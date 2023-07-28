#!/bin/bash
gunicorn --chdir /home/arpan/Desktop/pythonFiles/Hologram -w 3 --pid process.pid --bind 0.0.0.0:5000 --daemon wsgi:application --access-logfile logs/access.log --error-logfile logs/error.log
echo "Service Started!"