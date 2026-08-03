module.exports = {
  apps: [
    {
      name: 'phytograph',
      script: 'python3',
      args: '/home/user/webapp/backend/main.py',
      cwd: '/home/user/webapp',
      env: { PORT: 3000, PYTHONUNBUFFERED: '1' },
      watch: false,
      instances: 1,
      exec_mode: 'fork',
      error_file: '/home/user/webapp/logs/err.log',
      out_file: '/home/user/webapp/logs/out.log',
      log_date_format: 'YYYY-MM-DD HH:mm:ss'
    }
  ]
};
