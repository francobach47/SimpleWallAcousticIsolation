import sys

from PyQt5.QtWidgets import QApplication, QMainWindow, QTableWidgetItem, QCheckBox, QWidget, QVBoxLayout, QLabel, QPushButton, QMessageBox
from appMain import Ui_MainWindow
from matplotlib.backends.backend_qt5agg import (NavigationToolbar2QT as NavigationToolbar)

import numpy as np
import pandas as pd
import xlsxwriter

class MatplotlibWidget(QMainWindow):
    
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("Aislamiento a ruido aéreo de una pared simple mediante distintos métodos de predicción")
        
        # Push Buttons: 
        self.ui.pushButton_procesar.clicked.connect(self.procesar)        
        self.ui.pushButton_exportar.clicked.connect(self.exportar)
        self.ui.pushButton_borrar.clicked.connect(self.borrar)
        self.ui.pushButton_cerrar.clicked.connect(self.cerrar)
        
        self.addToolBar(NavigationToolbar(self.ui.MplWidget.canvas, self))

        # Se lee el archivo
        def lectura_archivo_excel(archivo):
            '''Levanta el archivo excel que queremos leer y procesar. 
               Pre: Debe ingresarse una cadena con el nombre del archivo, y el mismo 
               debe estar en el mismo directorio al código. Debe ser un archivo excel .xlsx
               Post: Devuelve un data frame con la tabla, independientemente cuál sea su 
               ubicación dentro del excel'''
            import pandas as pd
            df = pd.read_excel(archivo)
            for pos, i in enumerate(range(0, len(df)+1)):
                columnas = str(df.columns.to_list())
                if 'Material' in columnas:
                    df = pd.read_excel(archivo)
                    break
                fila_pos = df.iloc[pos].to_list()
                fila = ''.join(str(fila_pos))
                if 'nan' in fila:
                    if 'Material' in fila:
                        df = pd.read_excel(archivo, header = pos+1)
                        break
                    continue
            df.dropna(axis = 0, how = 'all', inplace = True) # Elimina filas vacias
            df.dropna(axis = 1, how = 'all', inplace = True) # Elimina columnas vacias
            return df
        
        self.df = lectura_archivo_excel('TABLA MATERIALES.xlsx') # Cargo el archivo
        
        # QComboBox
        self.ui.material.addItems([''] + list(self.df['Material'])) 
        
        #Deja la grafica en blanco desde el inicio y acomoda los ejes
        self.ui.MplWidget.canvas.axes.cla()
        self.ui.MplWidget.canvas.axes.set_xlabel('Frecuencia [Hz]')
        self.ui.MplWidget.canvas.axes.set_ylabel('R [dB]')
        freq_tercio = [20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160,
                       200, 250, 315, 400, 500, 630, 800, 1000, 1250,
                       1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000,
                       10000, 12500, 16000, 20000]
        self.ui.MplWidget.canvas.axes.plot(freq_tercio,[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],'white')
        self.ui.MplWidget.canvas.axes.set_xscale('log')
        self.ui.MplWidget.canvas.axes.set_xticks([20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160,
                                                  200, 250, 315, 400, 500, 630, 800, 1000, 1250,
                                                  1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000,
                                                  10000, 12500, 16000, 20000])
        self.ui.MplWidget.canvas.axes.set_xticklabels([20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160,
                                                       200, 250, 315, 400, 500, 630, 800, 1000, 1250,
                                                       1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000,
                                                       10000, 12500, 16000, 20000],rotation=45)
        self.ui.MplWidget.canvas.axes.set_yticks([0,10,20,30,40,50,60,70,80,90,100,110,120])
        
    def pared_simple(self): 
        ro0 = 1.18 # Densidad del aire [kg/m^3]
        c0 = 343 # Velocidad del sonido [m/s]
        freq_tercio = np.array([20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160,
                                200, 250, 315, 400, 500, 630, 800, 1000, 1250,
                                1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000,
                                10000, 12500, 16000, 20000])    
        espesor = float(self.ui.espesor.text())
        alto = float(self.ui.alto.text())
        largo = float(self.ui.largo.text())
        for pos, i in enumerate(self.df['Material'], start = 0):
                if i == self.ui.material.currentText():
                    ro = float(self.df.iloc[pos]['Densidad']) # Densidad 
                    E = float(self.df.iloc[pos]['Módulo de Young']) # Módulo de Young
                    eta = float(self.df.iloc[pos]['Factor de pérdidas']) # Factor de pérdidas
                    sigma = float(self.df.iloc[pos]['Módulo Poisson']) # Módulo de Poisson            
                    m = ro*(espesor) # Masa superficial del elemento [kg/m^2]
                    B = E*(espesor**3)/(12*(1 - (sigma)**2))  # Rigidez
                    fc = ((c0**2)/(2*np.pi))*(np.sqrt(m/B)) # Frecuencia crítica [Hz]
                    fd = (E/(2*np.pi*ro))*(np.sqrt(m/B)) # Frecuencia de densidad [Hz]
                    f11 = (c0**2/(4*fc))*((1/(alto)**2)+(1/(largo)**2))                           
        R_paredsimple = []                     
        for f in freq_tercio:
            if f < fc:
                R_paredsimple.append(round(20*np.log10(m*f) - 47, 1))
            elif f > fc and f < fd:
                R_paredsimple.append(round(20*np.log10(m*f) - 10*np.log10((np.pi)/(4*(eta + m/(485*np.sqrt(f))))) - 10*np.log10(fc/(f - fc)) - 47,1))
            elif f > fd:
                R_paredsimple.append(round(20*np.log10(m*f) - 47,1))
        R_paredsimple = np.array(R_paredsimple) 
        return R_paredsimple
        
    def sharp(self):
        ro0 = 1.18 # Densidad del aire [kg/m^3]
        c0 = 343 # Velocidad del sonido [m/s]
        freq_tercio = np.array([20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160,
                                200, 250, 315, 400, 500, 630, 800, 1000, 1250,
                                1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000,
                                10000, 12500, 16000, 20000])    
        espesor = float(self.ui.espesor.text())
        alto = float(self.ui.alto.text())
        largo = float(self.ui.largo.text())
        for pos, i in enumerate(self.df['Material'], start = 0):
                if i == self.ui.material.currentText():
                    ro = float(self.df.iloc[pos]['Densidad']) # Densidad 
                    E = float(self.df.iloc[pos]['Módulo de Young']) # Módulo de Young
                    eta = float(self.df.iloc[pos]['Factor de pérdidas']) # Factor de pérdidas
                    sigma = float(self.df.iloc[pos]['Módulo Poisson']) # Módulo de Poisson            
                    m = ro*(espesor) # Masa superficial del elemento [kg/m^2]
                    B = E*(espesor**3)/(12*(1 - (sigma)**2))  # Rigidez
                    fc = ((c0**2)/(2*np.pi))*(np.sqrt(m/B)) # Frecuencia crítica [Hz]
                    fd = (E/(2*np.pi*ro))*(np.sqrt(m/B)) # Frecuencia de densidad [Hz]
                    f11 = (c0**2/(4*fc))*((1/(alto)**2)+(1/(largo)**2))                        
        R_sharp = []
        interpolacion_posiciones = []
        interpolacion_frecuencias = []
        for pos, f in enumerate(freq_tercio, start = 0):                
            if f < 0.5*fc:
                R_sharp.append(round(10*np.log10(1 + ((np.pi*m*f)/(ro0*c0))**2 ) - 5.5,1))  
            if f >= 0.5*fc and f < fc:
                interpolacion_posiciones.append(pos)     
                interpolacion_frecuencias.append(f)
                R_sharp.append(0)
            elif f >= fc:
                (R1, R2) = (round(10*np.log10(1 + ((np.pi*m*f)/(ro0*c0))**2 ) + 10*np.log10((2*(eta + m/(485*np.sqrt(f)))*f)/(np.pi*fc)),1) , round(10*np.log10(1 + ((np.pi*m*f)/(ro0*c0))**2 ) - 5.5,1))
                R_sharp.append(min(R1, R2))
        # Se hace la interpolación lineal
        if len(interpolacion_frecuencias) >= 2:
            if interpolacion_frecuencias[0] > freq_tercio[0]:        
                (x1, x2) = (freq_tercio[interpolacion_posiciones[0] - 1], freq_tercio[interpolacion_posiciones[-1] + 1])         
                A = np.array(([x1, 1], [x2, 1]))
                b = np.array([R_sharp[interpolacion_posiciones[0] - 1], R_sharp[interpolacion_posiciones[-1] + 1]])
                a = np.linalg.solve(A, b)
                for i, pos in enumerate(interpolacion_posiciones):
                    R_sharp[pos] = round(a[0]*interpolacion_frecuencias[i] + a[1],1)
        R_sharp = np.array(R_sharp) 
        return R_sharp
        
    def davy(self):
        ro0 = 1.18 # Densidad del aire [kg/m^3]
        c0 = 343 # Velocidad del sonido [m/s]
        freq_tercio = np.array([20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160,
                                200, 250, 315, 400, 500, 630, 800, 1000, 1250,
                                1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000,
                                10000, 12500, 16000, 20000])    
        espesor = float(self.ui.espesor.text())
        alto = float(self.ui.alto.text())
        largo = float(self.ui.largo.text())
        for pos, i in enumerate(self.df['Material'], start = 0):
                if i == self.ui.material.currentText():
                    ro = float(self.df.iloc[pos]['Densidad']) # Densidad 
                    E = float(self.df.iloc[pos]['Módulo de Young']) # Módulo de Young
                    eta = float(self.df.iloc[pos]['Factor de pérdidas']) # Factor de pérdidas
                    sigma = float(self.df.iloc[pos]['Módulo Poisson']) # Módulo de Poisson            
                    m = ro*(espesor) # Masa superficial del elemento [kg/m^2]
                    B = E*(espesor**3)/(12*(1 - (sigma)**2))  # Rigidez
                    fc = ((c0**2)/(2*np.pi))*(np.sqrt(m/B)) # Frecuencia crítica [Hz]
                    fd = (E/(2*np.pi*ro))*(np.sqrt(m/B)) # Frecuencia de densidad [Hz]
                    f11 = (c0**2/(4*fc))*((1/(alto)**2)+(1/(largo)**2))  
        R_davy = []
        average = 3 # % promedio definido por Davy
        dB = 0.236
        octava = 3
        def Single_leaf_Davy(frecuencia, ro, E, sigma, espesor, eta, alto, largo):
            cos21Max = 0.9 # Ángulo límite definido en el trabajo de Davy
            densidad_superficie = ro*espesor 
            frecuencia_critica = np.sqrt(12*ro*(1-sigma**2)/E)*((c0**2)/(2*np.pi*espesor))
            normal = (ro0*c0)/(np.pi*frecuencia*densidad_superficie)
            normal2 = normal**2
            e = (2*largo*alto)/(largo + alto)
            cos2l = c0/(2*np.pi*frecuencia*e)
            if cos2l > cos21Max:
                cos2l = cos21Max
            tau1 = normal2*np.log((normal2 + 1)/(normal2 + cos2l))
            ratio = frecuencia/frecuencia_critica
            r = 1 - 1/ratio
            if r < 0:
                r = 0
            G = np.sqrt(r)
            rad = Sigma(G, frecuencia, alto, largo)
            rad2= rad**2
            netatotal = eta + rad*normal
            z = 2/netatotal
            y = np.arctan(z) - np.arctan(z*(1-ratio))
            tau2 = (normal2*rad2*y)/(netatotal*2*ratio)
            tau2 = tau2*shear(frecuencia, ro, E, sigma, espesor)
            if frecuencia < frecuencia_critica:
                tau = tau1 + tau2
            else:
                tau = tau2
            single_leaf = -10*np.log10(tau)
            return single_leaf
        def Sigma(G, frecuencia, alto, largo):
            w = 1.3
            beta = 0.234
            n = 2
            S = largo*alto
            U = 2*(largo + alto)
            twoa = 4*S/U
            k = (2*np.pi*frecuencia)/c0
            f = w*np.sqrt((np.pi)/(k*twoa))
            if f > 1:
                f = 1
            h = 1/(np.sqrt((k*twoa)/(np.pi))*2/3 - beta)
            q = (2*np.pi)/((k**2)*S)
            qn = q**n
            if G < f:
                alpha = h/f - 1
                xn = (h - alpha*G)**n
            else:
                xn = G**n
            rad = (xn + qn)**(-1/n)
            return rad
        def shear(frecuencia, ro, E, sigma, espesor):
            omega = 2*np.pi*frecuencia
            chi = ((1 + sigma)/(0.87 + 1.12*sigma))**2
            X = (espesor**2)/12
            QP = E/(1 - sigma**2)
            C = -(omega)**2
            B = C*(1 + 2*(chi/(1 - sigma)))*X
            A = X*QP/ro
            kbcor2 = (-B + np.sqrt(B**2 - 4*A*C))/(2*A)
            kb2 = np.sqrt(-C/A)
            G = E/(2*(1 + sigma))
            kT2 = -C*ro*chi/G
            kL2 = -C*ro/QP
            kS2 = kT2 + kL2
            ASl = (1 + X*(kbcor2*kT2/kL2 - kT2))**2
            BSl = 1 - X*kT2 + kbcor2*kS2/(kb2**2)  
            CSl = np.sqrt(1 - X*kT2 + (kS2**2)/(4*kb2**2))
            out = ASl/(BSl*CSl)
            return out
        for f in freq_tercio:
            eta_total = eta + m/(485*np.sqrt(f))
            ratio = f/fc
            limit = 2**(1/(2*octava))
            if (ratio < 1/limit) or (ratio > limit):
                TLost = Single_leaf_Davy(f, ro, E, sigma, espesor, eta_total, alto, largo)
                R_davy.append(round(TLost,1))
            else:
                Avsingle_leaf = 0
                for i in range(1, average+1):
                    factor = 2**((2*i - 1 - average)/(2*average*octava))
                    aux = 10**(-(Single_leaf_Davy(f*factor, ro, E, sigma, espesor, eta_total, alto, largo))/10)
                    Avsingle_leaf += aux
                TLost = -10*np.log10(Avsingle_leaf/average)
                R_davy.append(round(TLost,1))
        R_davy = np.array(R_davy)
        return R_davy

    def ISO(self):
        ro0 = 1.18 # Densidad del aire [kg/m^3]
        c0 = 343 # Velocidad del sonido [m/s]
        freq_tercio = np.array([20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160,
                                200, 250, 315, 400, 500, 630, 800, 1000, 1250,
                                1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000,
                                10000, 12500, 16000, 20000])    
        espesor = float(self.ui.espesor.text())
        alto = float(self.ui.alto.text())
        largo = float(self.ui.largo.text())
        for pos, i in enumerate(self.df['Material'], start = 0):
                if i == self.ui.material.currentText():
                    ro = float(self.df.iloc[pos]['Densidad']) # Densidad 
                    E = float(self.df.iloc[pos]['Módulo de Young']) # Módulo de Young
                    eta = float(self.df.iloc[pos]['Factor de pérdidas']) # Factor de pérdidas
                    sigma = float(self.df.iloc[pos]['Módulo Poisson']) # Módulo de Poisson            
                    m = ro*(espesor) # Masa superficial del elemento [kg/m^2]
                    B = E*(espesor**3)/(12*(1 - (sigma)**2))  # Rigidez
                    fc = ((c0**2)/(2*np.pi))*(np.sqrt(m/B)) # Frecuencia crítica [Hz]
                    fd = (E/(2*np.pi*ro))*(np.sqrt(m/B)) # Frecuencia de densidad [Hz]
                    f11 = (c0**2/(4*fc))*((1/(alto)**2)+(1/(largo)**2)) 
        R_iso = []
        if largo > alto: # siempre l1 > l2 
            l1 = largo
            l2 = alto
        else:
            l1 = alto
            l2 = largo 
        for f in freq_tercio: 
            def etatotal(f):
                eta_tot = eta + m/(485*np.sqrt(f))
                return eta_tot
            def delta1(f): 
                lamb = np.sqrt(f/fc)
                delta_1 = (((1 - lamb**2)*np.log((1+lamb)/(1-lamb)) + 2*lamb)/(4*(np.pi**2)*(1-lamb**2)**1.5))
                return delta_1
            def delta2(f):
                lamb = np.sqrt(f/fc)
                delta_2 = (8*(c0**2)*(1 - 2*lamb**2))/((fc**2)*(np.pi**4)*l1*l2*lamb*np.sqrt(1 - lamb**2))
                return delta_2
            def radforzada(f): 
                Lambda = - 0.964 - (0.5 + l2/(np.pi*l1))*np.log(l2/l1) + ((5*l2)/(2*np.pi*l1)) - (1/(4*np.pi*l1*l2*((2*np.pi*f)/c0)**2))
                sigma = 0.5*(np.log(((2*np.pi*f)/c0)*np.sqrt(l1*l2)) - Lambda) # Factor de radiación para ondas forzadas
                return sigma
            def sigma1(f):
                sigma_1 = 1/(np.sqrt(1 - fc/f)) 
                return sigma_1
            def sigma2(f):
                sigma_2 = 4*l1*l2*(f/c0)**2
                return sigma_2
            def sigma3(f):
                sigma_3 = np.sqrt((2*np.pi*f*(l1+l2))/(16*c0))
                return sigma_3
            
            if f > fc/2:
                delta_2 = 0
            else:
                delta_2 = delta2(f)
            # Calculamos el factor de radiación por ondas libres  
            if f11 <= fc/2: 
                if f >= fc:
                    rad_libre = sigma1(f)    
                elif f < fc:
                    lamb = np.sqrt(f/fc)
                    delta_1 = (((1 - lamb**2)*np.log((1+lamb)/(1-lamb)) + 2*lamb)/(4*(np.pi**2)*(1-lamb**2)**1.5))
                    delta_2 = (8*(c0**2)*(1 - 2*lamb**2))/((fc**2)*(np.pi**4)*l1*l2*lamb*np.sqrt(1 - lamb**2))
                    rad_libre = ((2*(l1 + l2)*c0*delta_1/(l1*l2*fc))) + delta_2
                sigma_2 = sigma2(f)
                if f<f11 and f<fc/2 and rad_libre > sigma_2:
                    rad_libre = sigma_2
            elif (f11 > fc/2):
                sigma_1 = sigma1(f)
                sigma_2 = sigma2(f)
                sigma_3 = sigma3(f)
                if (f < fc) and (sigma_2 < sigma_3):
                    rad_libre = sigma_2
                elif (f > fc) and (sigma_1 < sigma_3):
                    rad_libre = sigma_1
                else:
                    rad_libre = sigma_3
            if rad_libre > 2:
                rad_libre = 2 
            if f < fc:
                rad_forzada = radforzada(f)
                eta_total = etatotal(f)
                tao = abs((((2*ro0*c0)/(2*np.pi*f*m))**2)*(2*rad_forzada + (((l1 + l2)**2)/(l1**2 + l2**2))*np.sqrt(fc/f)*(rad_libre**2)/eta_total))
                R_iso.append(round(float(-10*np.log10(tao)),1)) 
            elif f == fc:
                eta_total = etatotal(f)
                tao = abs((((2*ro0*c0)/(2*np.pi*f*m))**2)*((np.pi*(rad_libre)**2)/(2*eta_total)))
                R_iso.append(round(float(-10*np.log10(tao)),1))
            elif f > fc:
                eta_total = etatotal(f)
                tao = abs((((2*ro0*c0)/(2*np.pi*f*m))**2)*((np.pi*fc*(rad_libre)**2)/(2*f*eta_total)))
                R_iso.append(round(float(-10*np.log10(tao)),1))
        R_iso = np.array(R_iso)
        return R_iso

    def procesar(self):
        ''' Método que se encarga del procesamiento y gráfico de los datos'''
        self.ui.MplWidget.canvas.axes.cla()
        freq_tercio = np.array([20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160,
                                200, 250, 315, 400, 500, 630, 800, 1000, 1250,
                                1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000,
                                10000, 12500, 16000, 20000])  
        try:
            errores = False
            # Primero vemos si ocurre un error de los detallados en el método errores
            if self.ui.espesor.text() == '' or self.ui.alto.text() == '' or self.ui.largo.text() == '':
                errores = True
            if self.ui.espesor.text() == '0' or self.ui.alto.text() == '0' or self.ui.largo.text() == '0':
                errores = True    
            def has_numbers(inputString):
                  return any(char.isdigit() for char in inputString)
            if (has_numbers(self.ui.espesor.text()) == False and self.ui.espesor.text() != '') or (has_numbers(self.ui.alto.text()) == False and self.ui.alto.text() != '') or (has_numbers(self.ui.largo.text()) == False and self.ui.largo.text() != ''):   
                errores = True    
            if self.ui.material.currentText() == '':
                errores = True
            if not self.ui.checkBox_sharp.isChecked() and not self.ui.checkBox_paredsimple.isChecked() and not self.ui.checkBox_davy.isChecked() and not self.ui.checkBox_ISO.isChecked():
                errores = True
            if (',' in self.ui.largo.text()) or (',' in self.ui.espesor.text()) or (',' in self.ui.alto.text()):
                errores = True
            if errores == True:
                self.errores() 
                
            self.ui.MplWidget.canvas.axes.grid()
            if self.ui.checkBox_paredsimple.isChecked():    
                R_paredsimple = self.pared_simple()
                self.ui.MplWidget.canvas.axes.plot(freq_tercio, R_paredsimple, 'red', label="Pared simple") 
                self.ui.MplWidget.canvas.axes.set_xscale('log')
                self.ui.MplWidget.canvas.axes.set_xticks([20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160,
                                                          200, 250, 315, 400, 500, 630, 800, 1000, 1250,
                                                          1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000,
                                                          10000, 12500, 16000, 20000])
                self.ui.MplWidget.canvas.axes.set_xticklabels([20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160,
                                                               200, 250, 315, 400, 500, 630, 800, 1000, 1250,
                                                               1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000,
                                                               10000, 12500, 16000, 20000],rotation=45)
                self.ui.MplWidget.canvas.axes.set_yticks([0,10,20,30,40,50,60,70,80,90,100,110,120])
                self.ui.MplWidget.canvas.axes.legend(loc = 'lower right')
                self.ui.MplWidget.canvas.axes.set_xlabel('Frecuencia [Hz]')
                self.ui.MplWidget.canvas.axes.set_ylabel('R [dB]')
                self.ui.MplWidget.canvas.draw()              

            if self.ui.checkBox_sharp.isChecked():
                R_sharp = self.sharp()
                self.ui.MplWidget.canvas.axes.plot(freq_tercio, R_sharp, 'blue', label="Sharp") 
                self.ui.MplWidget.canvas.axes.set_xscale('log')
                self.ui.MplWidget.canvas.axes.set_xticks([20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160,
                                                          200, 250, 315, 400, 500, 630, 800, 1000, 1250,
                                                          1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000,
                                                          10000, 12500, 16000, 20000])
                self.ui.MplWidget.canvas.axes.set_xticklabels([20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160,
                                                               200, 250, 315, 400, 500, 630, 800, 1000, 1250,
                                                               1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000,
                                                               10000, 12500, 16000, 20000],rotation=45)
                self.ui.MplWidget.canvas.axes.set_yticks([0,10,20,30,40,50,60,70,80,90,100,110,120])
                self.ui.MplWidget.canvas.axes.legend(loc = 'lower right')
                self.ui.MplWidget.canvas.axes.set_xlabel('Frecuencia [Hz]')
                self.ui.MplWidget.canvas.axes.set_ylabel('R [dB]')
                self.ui.MplWidget.canvas.draw()
            
            if self.ui.checkBox_davy.isChecked():
                R_davy = self.davy()
                self.ui.MplWidget.canvas.axes.plot(freq_tercio, R_davy, 'green', label="Davy") 
                self.ui.MplWidget.canvas.axes.set_xscale('log')
                self.ui.MplWidget.canvas.axes.set_xticks([20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160,
                                                          200, 250, 315, 400, 500, 630, 800, 1000, 1250,
                                                          1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000,
                                                          10000, 12500, 16000, 20000])
                self.ui.MplWidget.canvas.axes.set_xticklabels([20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160,
                                                               200, 250, 315, 400, 500, 630, 800, 1000, 1250,
                                                               1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000,
                                                               10000, 12500, 16000, 20000],rotation=45)
                self.ui.MplWidget.canvas.axes.set_yticks([0,10,20,30,40,50,60,70,80,90,100,110,120])
                self.ui.MplWidget.canvas.axes.legend(loc = 'lower right') 
                self.ui.MplWidget.canvas.axes.set_xlabel('Frecuencia [Hz]')
                self.ui.MplWidget.canvas.axes.set_ylabel('R [dB]')
                self.ui.MplWidget.canvas.draw()
                
            if self.ui.checkBox_ISO.isChecked():
                R_iso = self.ISO()
                self.ui.MplWidget.canvas.axes.plot(freq_tercio, R_iso, 'orange', label="ISO") 
                self.ui.MplWidget.canvas.axes.set_xscale('log')
                self.ui.MplWidget.canvas.axes.set_xticks([20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160,
                                                          200, 250, 315, 400, 500, 630, 800, 1000, 1250,
                                                          1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000,
                                                          10000, 12500, 16000, 20000])
                self.ui.MplWidget.canvas.axes.set_xticklabels([20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160,
                                                               200, 250, 315, 400, 500, 630, 800, 1000, 1250,
                                                               1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000,
                                                               10000, 12500, 16000, 20000],rotation=45)
                self.ui.MplWidget.canvas.axes.set_yticks([0,10,20,30,40,50,60,70,80,90,100,110,120])
                self.ui.MplWidget.canvas.axes.legend(loc = 'lower right')
                self.ui.MplWidget.canvas.axes.set_xlabel('Frecuencia [Hz]')
                self.ui.MplWidget.canvas.axes.set_ylabel('R [dB]')
                self.ui.MplWidget.canvas.draw()
                
        # Estas excepciones en verdad las atrapamos en los if del try        
        except ValueError: 
            None
        except ZeroDivisionError:
            None
        except UnboundLocalError:
            None
        
    def exportar(self):
        '''Método que se encarga de exportar los datos'''        
        try:
            # Calculamos fc
            freq_tercio = np.array([20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160,
                                    200, 250, 315, 400, 500, 630, 800, 1000, 1250,
                                    1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000,
                                    10000, 12500, 16000, 20000]) 
            c0 = 343 # Velocidad del sonido [m/s]   
            espesor = float(self.ui.espesor.text())
            alto = float(self.ui.alto.text())
            largo = float(self.ui.largo.text())
            Material = self.ui.material.currentText()
            for pos, i in enumerate(self.df['Material'], start = 0):
                if i == self.ui.material.currentText():
                    ro = float(self.df.iloc[pos]['Densidad']) # Densidad 
                    E = float(self.df.iloc[pos]['Módulo de Young']) # Módulo de Young
                    sigma = float(self.df.iloc[pos]['Módulo Poisson']) # Módulo de Poisson            
                    m = ro*(espesor) # Masa superficial del elemento [kg/m^2]
                    B = E*(espesor**3)/(12*(1 - (sigma)**2))  # Rigidez
                    fc = round(((c0**2)/(2*np.pi))*(np.sqrt(m/B)),2) # Frecuencia crítica [Hz]
                    
            # Definimos el data frame de la tabla con los datos            
            df_datos = pd.DataFrame()
            df_datos['Material'] = [Material]
            df_datos['Largo [m]'] = [float(self.ui.largo.text())]
            df_datos['Alto [m]'] = [float(self.ui.alto.text())]
            df_datos['Espesor [m]'] = [float(self.ui.espesor.text())]
            df_datos['Fc [Hz]'] = [fc]
            
            # Definimos el data frame de la tabla que contendrá la información de los distintos métodos
            df_metodos = pd.DataFrame()
            if self.ui.checkBox_paredsimple.isChecked():    
                df_paredsimple = pd.DataFrame()
                R_paredsimple = self.pared_simple()
                df_paredsimple['Método'] = ['Pared simple']
                for pos, i in enumerate(freq_tercio):
                    df_paredsimple[i] = R_paredsimple[pos]
                df_metodos = df_metodos.append(df_paredsimple)

            if self.ui.checkBox_sharp.isChecked():
                df_sharp = pd.DataFrame()
                R_sharp = self.sharp()
                df_sharp['Método'] = ['Sharp']
                for pos, i in enumerate(freq_tercio):
                    df_sharp[i] = R_sharp[pos]
                df_metodos = df_metodos.append(df_sharp)
                
            if self.ui.checkBox_davy.isChecked():
                df_davy = pd.DataFrame()
                R_davy = self.davy()
                df_davy['Método'] = ['Davy']
                for pos, i in enumerate(freq_tercio):
                    df_davy[i] = R_davy[pos]
                df_metodos = df_metodos.append(df_davy)
                
            if self.ui.checkBox_ISO.isChecked():
                df_iso = pd.DataFrame()
                R_iso = self.ISO()
                df_iso['Método'] = ['ISO 12354-1:2001']
                for pos, i in enumerate(freq_tercio):
                    df_iso[i] = R_iso[pos]        
                df_metodos = df_metodos.append(df_iso)
            
            # Generamos el excel y cambiamos ciertas configuraciones para una mejor presentación
            writer = pd.ExcelWriter(f'Resultados - {Material}.xlsx', engine='xlsxwriter')
            df_datos.to_excel(writer, sheet_name = 'Gráficos y valores de R', startrow = 1, startcol = 1, index = False) 
            df_metodos.to_excel(writer, sheet_name = 'Gráficos y valores de R', startrow = 5, startcol = 1, index = False)
            
            workbook = writer.book
            worksheet = writer.sheets['Gráficos y valores de R']
            
            # Agregamos bordes
            bordes = workbook.add_format({'bottom':1, 'top':1, 'left':1, 'right':1})
            worksheet.conditional_format( 'A1:BA50' , { 'type' : 'no_blanks' , 'format' : bordes} )
            
            # Pintamos los cabezales
            color = workbook.add_format({'bg_color' : '#FF6600'})
            worksheet.conditional_format('B2:F2', {'type' : 'no_blanks',
                                                   'format' : color})
            worksheet.conditional_format('B6:AG6', {'type' : 'no_blanks',
                                                    'format' : color})
           
            # Alineamos al centro
            cell_format = workbook.add_format({'align': 'center'})
            cell_format.set_align('hcenter')
            worksheet.set_column('A:AG',None,cell_format)
            
            # Ajustamos los anchos de las columnas
            worksheet.set_column('B:B', 15)
            worksheet.set_column('C:AG', 12)
            
            # Agregamos el gráfico
            chart = workbook.add_chart({'type' : 'line'})
            lista_metodos = df_metodos['Método'].to_list()
            # Tenemos que graficar los datos de df_metodos el cual va a tener tantos gráficos
            # como métodos de cáclulo de R activos 
            for i in range(1, len(df_metodos) + 1):
                 chart.add_series({'name': lista_metodos[i-1],
                                   'categories': ['Gráficos y valores de R',5,2,5,33],
                                   'values': f'=Gráficos y valores de R!$C${6+i}:$AG${6+i}'}) 
            worksheet.insert_chart('C12', chart)
            chart.set_title ({'name': 'Gráfico de R'})
            chart.set_x_axis({'name' : 'Frecuencia [Hz]'})
            chart.set_y_axis({'name' : 'Índice de reducción sonora R [dB]'})
            chart.height = 400
            chart.width = 1200
            writer.save()
            
        except ValueError:
            self.errores()
        except ZeroDivisionError:
            self.errores()
        except UnboundLocalError:
            self.errores()
        except KeyError:
            self.errores()

    def errores(self):
        '''Ventana gráfica que aparece cuando saltan errores'''
        msg = QMessageBox()
        msg.setWindowTitle('Error')
        msg.setText('Falta ingresar datos o hay datos erroneos.')
        
        # Armamos una lista con los posibles errores para luego darlos en detalle
        errores = []
        
        # Faltan datos de espesor-largo-alto
        if self.ui.espesor.text() == '':
            espesor_error = 'Falta el dato del "espesor"'
            errores.append(espesor_error)
        if self.ui.alto.text() == '':
            alto_error = 'Falta el dato del "alto"'
            errores.append(alto_error)    
        if self.ui.largo.text() == '':
            largo_error = 'Falta el dato del "largo"'    
            errores.append(largo_error) 

        # Errores de datos en espesor-largo-alto
        if self.ui.espesor.text() == '0':
            espesor_cero = 'El "espesor" no puede valer 0'
            errores.append(espesor_cero)
        if self.ui.alto.text() == '0':
            alto_cero = 'El "alto" no puede valer 0'
            errores.append(alto_cero)
        if self.ui.largo.text() == '0':
            largo_cero = 'El "largo" no puede valer 0'
            errores.append(largo_cero)
            
        def has_numbers(inputString):
            return any(char.isdigit() for char in inputString)
        
        if has_numbers(self.ui.espesor.text()) == False and self.ui.espesor.text() != '':
            espesor_erroneo_error = 'Tipo de dato erróneo en "espesor"'
            errores.append(espesor_erroneo_error)
        if ',' in self.ui.espesor.text():
            espesor_erroneo_error = 'Tipo de dato erróneo en "espesor"'
            errores.append(espesor_erroneo_error)
        
        if has_numbers(self.ui.alto.text()) == False and self.ui.alto.text() != '':
            alto_erroneo_error = 'Tipo de dato erróneo en "alto"'
            errores.append(alto_erroneo_error)        
        if ',' in self.ui.alto.text():
            alto_erroneo_error = 'Tipo de dato erróneo en "alto"'
            errores.append(alto_erroneo_error)
        
        if has_numbers(self.ui.largo.text()) == False and self.ui.largo.text() != '':
            largo_erroneo_error = 'Tipo de dato erróneo en "largo"'
            errores.append(largo_erroneo_error)  
        if ',' in self.ui.largo.text():
            largo_erroneo_error = 'Tipo de dato erróneo en "largo"'
            errores.append(largo_erroneo_error)
        
        # Faltan datos en la barra desplegable de materiales
        if self.ui.material.currentText() == '':
            material_error = 'Falta elegir un material'
            errores.append(material_error)
        
        # Falta seleccionar un método
        if not self.ui.checkBox_sharp.isChecked() and not self.ui.checkBox_paredsimple.isChecked() and not self.ui.checkBox_davy.isChecked() and not self.ui.checkBox_ISO.isChecked():
            metodo_error = 'Falta elegir un método de cálculo'
            errores.append(metodo_error)
                   
        errores = ' \n'.join(errores)

        msg.setIcon(QMessageBox.Warning)        
        msg.setInformativeText('''Recuerda ingresar números y no texto.\nNúmeros decimales con "." y no con ","''')
        msg.setDetailedText(errores)
        x = msg.exec_()

    def borrar(self):
        '''Establecemos la primer posición de la barra de materiales. Los otros casilleros
        se borran configurando en la GUI, desde Qtdesigner'''
        self.ui.material.setCurrentIndex(0)   
        
        # Borra la info de los graficos
        self.ui.MplWidget.canvas.axes.cla()
        self.ui.MplWidget.canvas.axes.set_xlabel('Frecuencia [Hz]')
        self.ui.MplWidget.canvas.axes.set_ylabel('R [dB]')
        freq_tercio = [20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160,
                       200, 250, 315, 400, 500, 630, 800, 1000, 1250,
                       1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000,
                       10000, 12500, 16000, 20000]
        self.ui.MplWidget.canvas.axes.plot(freq_tercio,[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],'white')
        self.ui.MplWidget.canvas.axes.set_xscale('log')
        self.ui.MplWidget.canvas.axes.set_xticks([20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160,
                                                  200, 250, 315, 400, 500, 630, 800, 1000, 1250,
                                                  1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000,
                                                  10000, 12500, 16000, 20000])
        self.ui.MplWidget.canvas.axes.set_xticklabels([20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160,
                                                       200, 250, 315, 400, 500, 630, 800, 1000, 1250,
                                                       1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000,
                                                       10000, 12500, 16000, 20000],rotation=45)
        self.ui.MplWidget.canvas.axes.set_yticks([0,10,20,30,40,50,60,70,80,90,100,110,120])
        self.ui.MplWidget.canvas.draw()

    def cerrar(self):
        '''Función para cerrar la GUI'''
        self.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)
    window = MatplotlibWidget()
    window.show()
    sys.exit(app.exec_())

