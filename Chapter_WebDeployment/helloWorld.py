##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##                    Hello World Web App
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%% import packages
import cherrypy

#%% Web application will be written as a Python class.
# Methods of the class will be used to respond to client requests
class HelloWorld(object):
    @cherrypy.expose
    def index(self):
        return "Hello world!"

#%% execution settings
cherrypy.config.update({'server.socket_host': '0.0.0.0'})

if __name__ == '__main__':
    cherrypy.quickstart(HelloWorld()) # when this script is executed, host HelloWorld app 
