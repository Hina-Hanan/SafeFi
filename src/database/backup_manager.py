"""
Database Backup Manager for SafeFi DeFi Risk Assessment Agent.

Provides automated backup, recovery, and monitoring capabilities for PostgreSQL.
"""

import asyncio
import subprocess
from typing import Dict, Any
from datetime import datetime, timedelta
from pathlib import Path
import gzip

from ..config.settings import get_settings
from ..utils.logger import get_logger, log_function_call, log_error_with_context


class BackupManager:
    """
    PostgreSQL backup and recovery management system.
    
    Handles automated backups, retention policies, and recovery procedures
    for the SafeFi database system.
    """
    
    def __init__(self):
        """Initialize BackupManager."""
        self.settings = get_settings()
        self.logger = get_logger("BackupManager")
        
        # Backup configuration
        self.backup_dir = Path("backups")
        self.backup_dir.mkdir(exist_ok=True)
        
        self.retention_days = 30
        self.max_backup_size_gb = 10
        
        # Find pg_dump executable
        self.pg_dump_path = self._find_pg_dump()
    
    def _find_pg_dump(self) -> str:
        """Find pg_dump executable path."""
        common_paths = [
            r"C:\Program Files\PostgreSQL\17\bin\pg_dump.exe",
            r"C:\Program Files\PostgreSQL\16\bin\pg_dump.exe",
            r"C:\Program Files\PostgreSQL\15\bin\pg_dump.exe",
            r"C:\Program Files\PostgreSQL\14\bin\pg_dump.exe"
        ]
        
        for path in common_paths:
            if Path(path).exists():
                self.logger.info(f"Found pg_dump at: {path}")
                return path
        
        # Fallback to PATH
        self.logger.warning("pg_dump not found in common paths, using PATH")
        return "pg_dump"
    
    async def create_full_backup(self, compress: bool = True) -> Dict[str, Any]:
        """
        Create full database backup.
        
        Args:
            compress: Whether to compress the backup file
            
        Returns:
            Dictionary containing backup results
        """
        log_function_call("BackupManager.create_full_backup", {"compress": compress})
        
        try:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"safefi_full_backup_{timestamp}.sql"
            temp_backup_path = self.backup_dir / backup_filename
            
            # Build pg_dump command
            cmd = [
                self.pg_dump_path,
                "--host", self.settings.db_host,
                "--port", str(self.settings.db_port),
                "--username", self.settings.db_user,
                "--dbname", self.settings.db_name,
                "--verbose",
                "--clean",
                "--no-owner",
                "--no-privileges"
            ]
            
            # Set environment variable for password
            env = {"PGPASSWORD": self.settings.db_password}
            
            # Execute backup
            self.logger.info(f"Starting full backup to {temp_backup_path}")
            
            # Create uncompressed backup first
            with open(temp_backup_path, 'w') as f:
                result = subprocess.run(cmd, stdout=f, env=env, check=True, text=True, shell=True)
            
            # Compress using Python's gzip if requested
            if compress:
                compressed_path = temp_backup_path.with_suffix('.sql.gz')
                with open(temp_backup_path, 'rb') as f_in:
                    with gzip.open(compressed_path, 'wb') as f_out:
                        f_out.writelines(f_in)
                
                # Remove uncompressed file
                temp_backup_path.unlink()
                backup_path = compressed_path
            else:
                backup_path = temp_backup_path
            
            # Get backup file info
            backup_size = backup_path.stat().st_size
            
            backup_info = {
                'success': True,
                'backup_path': str(backup_path),
                'backup_size_bytes': backup_size,
                'backup_size_mb': round(backup_size / (1024 * 1024), 2),
                'compressed': compress,
                'created_at': datetime.utcnow().isoformat(),
                'database': self.settings.db_name
            }
            
            self.logger.info(f"Backup completed successfully: {backup_size / (1024*1024):.2f} MB")
            return backup_info
            
        except subprocess.CalledProcessError as e:
            error_msg = f"pg_dump failed with return code {e.returncode}"
            log_error_with_context(Exception(error_msg), {"compress": compress})
            return {
                'success': False,
                'error': error_msg,
                'created_at': datetime.utcnow().isoformat()
            }
        except Exception as e:
            log_error_with_context(e, {"compress": compress})
            return {
                'success': False,
                'error': str(e),
                'created_at': datetime.utcnow().isoformat()
            }
    
    async def restore_from_backup(self, backup_path: str) -> Dict[str, Any]:
        """
        Restore database from backup file.
        
        Args:
            backup_path: Path to backup file
            
        Returns:
            Dictionary containing restore results
        """
        log_function_call("BackupManager.restore_from_backup", {"backup_path": backup_path})
        
        try:
            backup_file = Path(backup_path)
            
            if not backup_file.exists():
                raise FileNotFoundError(f"Backup file not found: {backup_path}")
            
            # Build psql restore command
            cmd = [
                "psql",
                "--host", self.settings.db_host,
                "--port", str(self.settings.db_port),
                "--username", self.settings.db_user,
                "--dbname", self.settings.db_name,
                "--quiet"
            ]
            
            env = {"PGPASSWORD": self.settings.db_password}
            
            self.logger.info(f"Starting database restore from {backup_path}")
            
            # Handle compressed vs uncompressed files
            if backup_path.endswith('.gz'):
                # Restore from compressed backup
                with gzip.open(backup_file, 'rt') as f:
                    result = subprocess.run(cmd, stdin=f, env=env, check=True, text=True,
                                          capture_output=True, shell=True)
            else:
                # Restore from uncompressed backup
                with open(backup_file, 'r') as f:
                    result = subprocess.run(cmd, stdin=f, env=env, check=True, text=True,
                                          capture_output=True, shell=True)
            
            restore_info = {
                'success': True,
                'backup_path': backup_path,
                'restored_at': datetime.utcnow().isoformat(),
                'database': self.settings.db_name
            }
            
            self.logger.info("Database restore completed successfully")
            return restore_info
            
        except subprocess.CalledProcessError as e:
            error_msg = f"Database restore failed with return code {e.returncode}"
            if e.stderr:
                error_msg += f": {e.stderr.decode()}"
            
            log_error_with_context(Exception(error_msg), {"backup_path": backup_path})
            return {
                'success': False,
                'error': error_msg,
                'restored_at': datetime.utcnow().isoformat()
            }
        except Exception as e:
            log_error_with_context(e, {"backup_path": backup_path})
            return {
                'success': False,
                'error': str(e),
                'restored_at': datetime.utcnow().isoformat()
            }
    
    async def cleanup_old_backups(self) -> Dict[str, Any]:
        """
        Clean up old backup files based on retention policy.
        
        Returns:
            Dictionary containing cleanup results
        """
        log_function_call("BackupManager.cleanup_old_backups", {})
        
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=self.retention_days)
            
            removed_files = []
            total_size_freed = 0
            
            for backup_file in self.backup_dir.glob("safefi_*.sql*"):
                if backup_file.is_file():
                    file_mod_time = datetime.fromtimestamp(backup_file.stat().st_mtime)
                    
                    if file_mod_time < cutoff_date:
                        file_size = backup_file.stat().st_size
                        backup_file.unlink()
                        
                        removed_files.append({
                            'filename': backup_file.name,
                            'size_bytes': file_size,
                            'modified_at': file_mod_time.isoformat()
                        })
                        total_size_freed += file_size
            
            cleanup_info = {
                'success': True,
                'files_removed': len(removed_files),
                'total_size_freed_mb': round(total_size_freed / (1024 * 1024), 2),
                'retention_days': self.retention_days,
                'removed_files': removed_files,
                'cleaned_at': datetime.utcnow().isoformat()
            }
            
            self.logger.info(f"Cleanup completed: {len(removed_files)} files removed, "
                           f"{cleanup_info['total_size_freed_mb']} MB freed")
            
            return cleanup_info
            
        except Exception as e:
            log_error_with_context(e, {})
            return {
                'success': False,
                'error': str(e),
                'cleaned_at': datetime.utcnow().isoformat()
            }
    
    async def get_backup_status(self) -> Dict[str, Any]:
        """
        Get current backup system status and statistics.
        
        Returns:
            Dictionary containing backup status information
        """
        try:
            backup_files = list(self.backup_dir.glob("safefi_*.sql*"))
            
            if not backup_files:
                return {
                    'total_backups': 0,
                    'latest_backup': None,
                    'total_size_mb': 0,
                    'oldest_backup': None,
                    'status': 'no_backups'
                }
            
            # Get file stats
            file_stats = []
            total_size = 0
            
            for backup_file in backup_files:
                stat = backup_file.stat()
                file_stats.append({
                    'filename': backup_file.name,
                    'size_bytes': stat.st_size,
                    'created_at': datetime.fromtimestamp(stat.st_ctime).isoformat(),
                    'modified_at': datetime.fromtimestamp(stat.st_mtime).isoformat()
                })
                total_size += stat.st_size
            
            # Sort by modification time
            file_stats.sort(key=lambda x: x['modified_at'], reverse=True)
            
            status_info = {
                'total_backups': len(backup_files),
                'latest_backup': file_stats[0] if file_stats else None,
                'oldest_backup': file_stats[-1] if file_stats else None,
                'total_size_mb': round(total_size / (1024 * 1024), 2),
                'backup_directory': str(self.backup_dir),
                'retention_days': self.retention_days,
                'status': 'healthy' if file_stats else 'no_backups',
                'all_backups': file_stats
            }
            
            return status_info
            
        except Exception as e:
            log_error_with_context(e, {})
            return {
                'status': 'error',
                'error': str(e)
            }
    
    async def schedule_automated_backups(self, interval_hours: int = 24) -> None:
        """
        Schedule automated backup creation.
        
        Args:
            interval_hours: Hours between automated backups
        """
        try:
            self.logger.info(f"Starting automated backup scheduler (every {interval_hours} hours)")
            
            while True:
                # Create backup
                backup_result = await self.create_full_backup(compress=True)
                
                if backup_result['success']:
                    self.logger.info("Automated backup completed successfully")
                    
                    # Clean up old backups
                    cleanup_result = await self.cleanup_old_backups()
                    if cleanup_result['success']:
                        self.logger.info("Automated cleanup completed")
                else:
                    self.logger.error(f"Automated backup failed: {backup_result.get('error')}")
                
                # Wait for next backup
                await asyncio.sleep(interval_hours * 3600)
                
        except Exception as e:
            log_error_with_context(e, {"interval_hours": interval_hours})
