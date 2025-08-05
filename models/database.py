import MySQLdb

def init_db(app):
    def get_db_connection():
        return MySQLdb.connect(
            host=app.config['MYSQL_HOST'],
            user=app.config['MYSQL_USER'],
            passwd=app.config['MYSQL_PASSWORD'],
            db=app.config['MYSQL_DB']
        )
    app.config['get_db_connection'] = get_db_connection