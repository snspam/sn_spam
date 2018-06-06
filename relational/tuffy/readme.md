### Tuffy Installation ###

To get Tuffy up and running, follow the steps below:

1. Download the `tuffy.jar` file from the [Tuffy homepage](http://i.stanford.edu/hazy/tuffy/doc/). Place the file in this directory.
2. Download PostgreSQL version 8.4 or later from the [PostgreSQL homepage](https://www.postgresql.org/
). Follow install instructions below.
3. Fill in the `tuffy.conf` file with the necessary missing information.


### PostgreSQL Installation Info

Steps to setup a cluster instance and database within that instance.

1. `initdb <instance>`:
Creates a postgresql cluster! Create one if there isn't one already in `/usr/local/pgsql`

2. `pg_ctl -D /usr/local/pgsql/data -l /usr/local/pgsql/data/logfile start`:
Starts the cluster instance server.

3. `createdb <name>`:
Creates a db once the instance has been started.

You should now be able to connect to the created database.

Helpful tools and commands:

* *pg_ctl*: Helper tool for starting and stopping cluster instances. Does Postmaster tasks.

* *psql*: Helper tool for running sql commands for databases inside the running instance.

* `psql <db>`: Starts a CLI with the current instance and connects to the database. Using the CLI, you can create users, databases, alter roles, etc.

* `ps aux | grep postgres`: Useful for finding out if an instance is already running.