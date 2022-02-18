##%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
##                    Hello World Web App
## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%% import packages
import cherrypy

#%% FDD tool Web application
class FDDapp(object):
    @cherrypy.expose
    def getResults(self):
        processState = runPCAmodel() # returns 'All good' or 'Issue detected'
        return processState

#%% execution settings
cherrypy.config.update({'server.socket_host': '0.0.0.0'})

if __name__ == '__main__':
    cherrypy.quickstart(FDDapp()) # when this script is executed, host FDDapp app 
