#!/bin/bash

# Environment configuration
VENV_DIR="venv"
REQ_FILE="requirements.txt"

# Colors for console output
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

# Function to initialize the virtual environment
init_env() {
    if [ -d "$VENV_DIR" ]; then
        # Detect Windows (Scripts) vs Linux/Mac (bin)
        if [ -f "$VENV_DIR/Scripts/activate" ]; then
            source $VENV_DIR/Scripts/activate
        elif [ -f "$VENV_DIR/bin/activate" ]; then
            source $VENV_DIR/bin/activate
        fi
    else
        echo -e "${RED}Virtual environment does not exist! Run first: $0 env_init${NC}"
        exit 1
    fi
}

case "$1" in
    run)
        echo -e "${GREEN}Starting the development server...${NC}"
        init_env
        python manage.py runserver
        ;;

    env_init)
        echo -e "${GREEN}Initializing a new virtual environment...${NC}"
        # Using python (or python3 depending on the system)
        python -m venv $VENV_DIR

        # Activate the environment
        if [ -f "$VENV_DIR/Scripts/activate" ]; then
            source $VENV_DIR/Scripts/activate
        elif [ -f "$VENV_DIR/bin/activate" ]; then
            source $VENV_DIR/bin/activate
        fi

        # Install dependencies
        python -m pip install --upgrade pip wheel
        if [ -f "$REQ_FILE" ]; then
            pip install -r $REQ_FILE
        else
            echo -e "${RED}File $REQ_FILE not found!${NC}"
        fi
        ;;

    env)
        echo -e "${GREEN}Starting bash session in the virtual environment...${NC}"
        if [ -f "$VENV_DIR/Scripts/activate" ]; then
            /bin/bash -c ". $VENV_DIR/Scripts/activate; exec /bin/bash --norc -i"
        else
            /bin/bash -c ". $VENV_DIR/bin/activate; exec /bin/bash --norc -i"
        fi
        ;;

    lint)
        echo -e "${GREEN}Running flake8: https://www.flake8rules.com/${NC}"
        init_env
        shift
        # Default to current directory if no path is provided
        scan_path=${1:-"."}
        flake8 --statistics $scan_path
        ;;

    black)
        echo -e "${GREEN}Running black formatter...${NC}"
        init_env
        shift
        # Format main python directories in the Weka 2.0 project
        black data ml preprocessing register core Weka2_0 "$@"
        ;;

    db_init)
        echo -e "${GREEN}Preparing the database...${NC}"
        init_env
        echo -e "${GREEN}-> Creating migrations...${NC}"
        python manage.py makemigrations
        echo -e "${GREEN}-> Applying migrations...${NC}"
        python manage.py migrate
        echo -e "${GREEN}-> Seeding the database (seed_data)...${NC}"
        python manage.py seed_data
        echo -e "${GREEN}-> Creating a superuser (admin)...${NC}"
        python manage.py createsuperuser
        ;;

    manage)
        echo -e "${GREEN}Running django command: $@${NC}"
        shift
        init_env
        python manage.py "$@"
        ;;

    *)
        echo "Available commands:"
        echo -e "${RED}$0 run${NC}       - starts the development server (runserver)"
        echo -e "${RED}$0 env_init${NC}  - creates a new venv and installs requirements.txt"
        echo -e "${RED}$0 env${NC}       - starts a new console session with venv enabled"
        echo -e "${RED}$0 lint${NC}      - runs flake8 to check for PEP8 compliance"
        echo -e "${RED}$0 black${NC}     - runs black code formatter"
        echo -e "${RED}$0 db_init${NC}   - executes makemigrations, migrate, seed_data and createsuperuser"
        echo -e "${RED}$0 manage${NC}    - allows running any manage.py command (e.g. ./ctl manage shell)"
        ;;
esac