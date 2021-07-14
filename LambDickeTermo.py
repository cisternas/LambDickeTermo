import os, math
import numpy as np
import pandas as pd
from scipy import constants
from scipy.signal import find_peaks as fp
from scipy.stats import binom
from scipy.optimize import curve_fit
from scipy.integrate import simps

# Constantes

hbar = constants.value('Planck constant over 2 pi')
k_bolt = constants.value('Boltzmann constant')

try:
    from tqdm import trange
    tRange = trange
except ImportError:
    print("\033[1mSugerencia (tqdm)\033[0m \n")
    print("No se encontró en módulo 'tqdm' instalado.", "Dado que algunos procesos de este termómetro pueden demorar unos minutos",
    "recomendamos instalar dicho módulo para visualizar el estado de los mismos.",
    "\nPara instalarlo utilizar el comando 'pip install tqdm'. Luego volver a importar este termómetro.",
    "\nPara más información ingresar a https://pypi.org/project/tqdm/")
    print('\n')

    def tRange(n, desc='texto'):
        print(desc)
        return range(n)


########################################################################################################################################


class LambDickeTermo:

    def __init__(self, w_osc, lamb_dicke_eta, acoplamiento=1, tasa_decoherencia=0, pob_error=1e-5, med_max=300, 
                inicializar=False, error_relativo_tiempo=None):

        # Argumentos del termómetro
        self.w = float(w_osc)
        self.eta = float(lamb_dicke_eta)
        self.gamma = float(tasa_decoherencia/acoplamiento)
        self.omega = float(acoplamiento)
        self.pob_error = float(pob_error)
        self.med_max = int(med_max)
        self.error_t_rel = error_relativo_tiempo

        # Definición de temperatura y tiempo
        self.temp = self.rango_temp()
        self.time = self.rango_tiempo()

        # Path de archivos
        self.carpetas()
        self.archivos()
        self.inicio_carpetas()

        #### Creación del termómetro y carga de datos

        self.inicio_poblaciones()

        if hasattr(self, 'poblaciones') == True:
            if os.path.isfile(os.path.join(self.paths['general'], 'inicializacion_'+self.archivos_nombres['csv'])) and inicializar==False:
                print('\n\033[1mCargando inicialización previa\033[0m \n')
                self.cargar_inicializacion()
            else:
                print('\n\033[1mInicializando el termómetro\033[0m \n')
                self.inicializar()





########################################################################################################################################



    def help(self):
        print("Este objeto representa un estimador de la temperatura de un ion atrapado")
        print("Para realizar esta estimación se debe ingresar la frecuencia del oscilador (ion atrapado) w junto al factor de decoherencia gamma")
        print("Los valores ingresados son w = ",self.w," , gamma = ",self.gamma," y el parámetro de Lamb Dicke ",self.eta)
        print("Para estimar la temperatura del ion se deberá medir la probabilidad de encontrarlo en su estado excitado",
            "para los tiempos dados en 'self.tiempos_de_medicion()'. Si se conoce aproximadamente la temperatura a la cual",
            "se encuentra el ion, recomendamos utilizar la función 'self.rango_temperatura(T_min,T_max)' primero, proveyendo",
            "el intervalo [T_min, T_max] correspondiente.\nUna vez medidas las probabilidades, utilizar la función",
            "'self.estimar(prob, N)' para obtener la estimación. Esta función tiene como input las probabilidades medidas 'prob'",
            "además de la cantidad de mediciones realizadas 'N' en cada tiempo. Ambas listas deben ingresarse como listas o ",
            "arrays (numpy), cuyo orden debe ser el mismo que los tiempos de medición. La estimación tiene dos outputs:",
            "la temperatura estimada y el error asociado.")



########################################################################################################################################


    def inicio_carpetas(self):
        for key in self.paths.keys():
            path = self.paths[key]
            if not os.path.exists(path):
                os.makedirs(path)


    def archivos(self):
        self.archivos_nombres = {'numpy_pob': '_'.join( [str(self.eta), str(self.w)] )+'.npy',
                                 'numpy': '_'.join( [str(self.eta), str(self.w), str(self.gamma)] )+'.npy',
                                 'csv': '_'.join( [str(self.eta), str(self.w), str(self.gamma)] )+'.csv'
                                }

    def carpetas(self):
        self.paths = {'poblaciones': os.path.join(os.getcwd(), "data", "poblaciones"),
                      'fit': os.path.join(os.getcwd(), "data", "fit"),
                      'errores': os.path.join(os.getcwd(), "data", "errores"),
                      'general': os.path.join(os.getcwd(), "data", "general")
                     }


########################################################################################################################################


    def N_cota(self, T):
        """
        Cota superior en número de fonones para calcular la problación de un canal. El error relativo tolerado entre un término
        y su siguiente es "error".
        """
        x = np.exp(-hbar*self.w/(k_bolt*T))
        return int(np.ceil(np.log(self.pob_error/(1-x+self.pob_error))/np.log(x) - 1))


    def rabi_frec(self, n, m):
        """
        Frecuencia de rabi para 'n' fonones y el canal de interacción 'm'.
        """
        if m == 0:
            w = 1 - n*self.eta**2
        elif m == 1:
            w = np.sqrt(n+1)*self.eta
        elif m == -1:
            w = np.sqrt(n)*self.eta
        else:
            print('Error: Debe ingresarse para m alguno de los valores (-1,0,1)')
        return w


    def lamb_dicke_temp(self):
        """
        Temperatura de Lamb Dicke, definida a partir de la condición de Lamb Dicke. 
        """
        T = hbar*self.w / (k_bolt * np.log((1+self.eta**2)/(1-self.eta**2)))
        return T


    def rango_temp(self):
        """
        Definición del rango de temperaturas en base a lamb_dicke_temp() y la temperatura T_os asociada al oscilador armónico.
        """
        T_max = self.lamb_dicke_temp()
        T_min = hbar*self.w / (np.log(100)*k_bolt)
        return np.logspace(np.log10(T_min), np.log10(T_max), 6000)

    
    def rango_tiempo(self):
        """
        Definición del rango de tiempos, comenzando en 0 y finalizando en pi/eta**2.
        """
        return np.linspace(0,np.pi/self.eta**2, 6000)


    def transf_tiempo(self, t):
        """
        Transformación del tiempo Theta a t mediante la constante de acomplamiento Omega.
        """
        return t/self.omega

    def error_relativo(self):
        e_rel = np.logspace(-3,0,1000)
        return e_rel


########################################################################################################################################


    def inicializar(self):
        '''
        Inicialización del termómetro posterior a la creación de los archivos de población para cada canal. En orden, esta función 
            - Aplica la decoherencia a las poblaciones
            - Selecciona las campanas de información elegiendo la tolerancia óptima
            - Calcula los ajustes de cada población seleccionada
            - Calcula el error sistemático y estadístico del método
            - Guarda los resultados

        Esta función se iniciará solo si se está utilizando el termémtro por primera vez. En caso contrario, se deberá ingresar el 
        parámetro 'inicializar = True' al llamar al objeto LambDickeTermo para iniciarlo nuevamente.
        '''

        self.decoherencia(self.gamma)
        self.selector_campana(tol_opt=True)
        tol = self.tolerancia_opt[self.tolerancia_opt['opt']==True]['tol']
        self.fit()

        self.fit_info.to_csv(os.path.join(self.paths['fit'], self.archivos_nombres['csv']), index=False)

        self.error_sistematico()
        self.error_estadistico()

        df = pd.DataFrame({'eta': self.eta, 'decoherencia': self.gamma*self.omega, 'w_osc':self.w, 'tolerancia':tol,
                           'T_sistematico': self.T_S, 'T_lambDicke': max(self.temp), 'error_sistematico':self.sist_max,
                           'cobertura':self.cobertura})

        df.to_csv(os.path.join(self.paths['general'], 'inicializacion_'+self.archivos_nombres['csv']), index=False)
        np.save(os.path.join(self.paths['errores'], self.archivos_nombres['numpy']), self.n_estadistico)

    
    def cargar_inicializacion(self):
        '''
        Carga de la inicialización previa. Habiendo cargado los archivos de población, esta función
            - Aplica la decoherencia a las poblaciones
            - Carga las curvas de información
            - Carga los archivos generados por la inicialización previa

        Esta función de inicia si existen archivos de una inicialización previa. Si se desea volver a inicializar el termómetro
        mediante la función 'self.inicializar()' se deberá ingresar el parámetro 'inicializar=True' al cargar el objeto LambDickeTermo
        '''

        self.decoherencia(self.gamma)
        self.cargar_infos()
        self.fit_info = pd.read_csv(os.path.join(self.paths['fit'], self.archivos_nombres['csv']))
        self.fit_funciones()
        inicializacion = pd.read_csv(os.path.join(self.paths['general'], 'inicializacion_'+self.archivos_nombres['csv']))
        self.T_S = inicializacion['T_sistematico']
        self.sist_max = inicializacion['error_sistematico']
        self.cobertura = inicializacion['cobertura']
        self.n_estadistico = np.load(os.path.join(self.paths['errores'], self.archivos_nombres['numpy']))



########################################################################################################################################




    def inicio_poblaciones(self): 
        '''
        Carga de los archivos de población. En el caso de no existir dichos archivos para los parámetros ingresados se da la opción
        de crearlos.
        '''

        if os.path.isfile(os.path.join(self.paths['poblaciones'], 'carrier_'+self.archivos_nombres['numpy_pob'])):
            print("Archivos encontrados. Cargando poblaciones")
            carrier = np.load(os.path.join(self.paths['poblaciones'],'carrier_'+self.archivos_nombres['numpy_pob']))
            print("Población carrier cargada")
            blue = np.load(os.path.join(self.paths['poblaciones'],'blue_'+self.archivos_nombres['numpy_pob']))
            print("Población blue sideband cargada")
            red = np.load(os.path.join(self.paths['poblaciones'],'red_'+self.archivos_nombres['numpy_pob']))
            print("Población red sideband cargada")

            self.poblaciones = {'carrier':carrier, 'blue':blue, 'red':red}
            print("Proceso finalizado.")
            self.p = self.poblaciones

        else:
            print("Archivos no encontrados en",self.paths['poblaciones'])
            print("¿Desea crearlos? (y/n)")
            opcion = input()
            while opcion.lower() not in ["y","n"]:
                print("Valor ingresado incorrecto. Intente nuevamente.")
                print("¿Desea crear los archivos de poblaciones? (y/n)")
                opcion = input()

            if opcion.lower() == "y":
                self.crear_poblaciones()

            elif opcion.lower() == "n":
                print("Los archivos de poblaciones no serán creados. Proceso finalizado")


                          

    def cargar_poblaciones(self, lamb_dicke, w_osc, drop_canal=['blue']):
        '''
        Función para cargar los archivos de población. Es posible elegir la opción de no cargar un canal ingresando su nombre
        ('carrier', 'blue' o 'red') dentro de una lista en el parametro 'drop_canal'.

        Esta función solo carga las poblaciones. Para obtener la selección de campanas y los ajustes a las poblaciones seleccionadas
        se deberá utilizar las funciones 'self.selector_campana' y 'self.fit'. Recordar que, en el caso de tener decoherencia, se deberá
        disparar la función 'self.decoherencia' antes de la selección. 
        '''

        self.eta = float(lamb_dicke)
        self.w = float(w_osc)
        self.time = self.rango_tiempo()
        self.temp = self.rango_temp()
        
        self.archivos()

        poblaciones = {}

        try:
            print("Cargando poblaciones")
            if 'carrier' not in drop_canal:
                carrier = np.load(os.path.join(self.paths['poblaciones'],'carrier_'+self.archivos_nombres['numpy']))
                print("Población carrier cargada")
                poblaciones = {**poblaciones, **{'carrier':carrier}}
            if 'blue' not in drop_canal:
                blue = np.load(os.path.join(self.paths['poblaciones'],'blue_'+self.archivos_nombres['numpy']))
                print("Población blue sideband cargada")
                poblaciones = {**poblaciones, **{'blue':blue}}
            if 'red' not in drop_canal:
                red = np.load(os.path.join(self.paths['poblaciones'],'red_'+self.archivos_nombres['numpy']))
                print("Población red sideband cargada")
                poblaciones = {**poblaciones, **{'red':red}}

            self.poblaciones = poblaciones
            self.p = self.poblaciones
            print("Poblaciones cargadas")

        except:
            print("Archivos no encontrados")


    def cargar_infos(self):
        '''
        Función para calcular las curvas de información en base a los archivos de población. En el caso de tener decoherencia
        se deberá disparar la función 'self.decoherencia' previamente.

        Las curvas de información se almacenarán en el atributo 'self.infos'.
        '''

        import warnings
        warnings.filterwarnings('ignore')

        infos = {}
        print('Cargando información de Fisher')
        for key in self.poblaciones.keys():
            info = self.calcular_info(self.poblaciones[key])
            infos = {**infos, **{key:info}}

        self.infos = infos


    def reset(self):
        self.poblaciones = self.p
        self.gamma = 0
        self.archivos()



########################################################################################################################################

    def func_poblaciones(self, canal, t, T):
        """
        Función que devuelve el valor de la población 'canal' para tiempo 't' y temperatura 'T'.
        """

        t, T = np.atleast_1d(t), np.atleast_1d(T)

        if canal not in ['carrier','blue','red']:
            return "El canal debe coincidir con 'carrier', 'blue' o 'red'"

        if canal == 'carrier':
            x = np.exp(-hbar*self.w/(k_bolt*T))
            o = np.ones((len(t)))
            p = (np.outer(np.cos(t),1-x)-np.outer(np.cos((1+self.eta**2)*t),x*(1-x)))/(1-2*np.outer(np.cos(self.eta**2*t),x)+np.outer(o,x**2))
            y = (1/2)*(1-p)
        
        else:
            m = {'blue':1, 'red':-1}[canal]
            N_max = self.N_cota(max(T))
            x = np.exp(-hbar*self.w / (k_bolt*T))
            X = (np.power(np.outer(x,np.ones(N_max+1)), range(N_max+1)).T)*(1-x)
            sin = np.sin(np.outer(t,self.rabi_frec(np.asarray(range(N_max+1)),m))/2)**2
            y = np.matmul(sin,X)

        return np.squeeze(y)


    def func_poblaciones_error(self, canal, t, T, sigma_t):
        '''
        Función que calcula el valor de la población 'canal' para tiempo 't' y temperatura 'T' teniendo en cuenta
        errores en los tiempos de medición. El error se ingresa en el parámetro 'sigma_t'.
        '''

        t, T = np.atleast_1d(t), np.atleast_1d(T)

        if canal not in ['carrier','blue','red']:
            return "El canal debe coincidir con 'carrier', 'blue' o 'red'"

        m = {'carrier':0, 'blue':1, 'red':-1}[canal]
        N_max = self.N_cota(max(T))
        x = np.exp(-hbar*self.w / (k_bolt*T))
        X = (np.power(np.outer(x,np.ones(N_max+1)), range(N_max+1)).T)*(1-x)
        rabi = self.rabi_frec(np.asarray(range(N_max+1)),m)
        cos = np.cos(np.outer(t,rabi))
        exp = np.exp(-1/2 * (sigma_t*rabi)**2)

        y = np.matmul((1-cos*exp)/2, X)

        return np.squeeze(y)





    def crear_poblaciones(self):
        """
        A partir de un error relativo "error" para la función N_cota, esta función cálcula y guarda el valor de las poblaciones carrier
        y blue/red sideband para 4000 valores de temperatura (entre 1e-5K y 1e-1K) y 6000 valores de tiempo entre 0s y  2*pi/eta**2 s
        (periodo de la envolvente del carrier).
        """

        self.time = self.rango_tiempo()
        self.temp = self.rango_temp()
        x = np.exp(-hbar*self.w/(k_bolt*self.temp))

        print('Calculando poblaciones')

        carrier = self.func_poblaciones('carrier', self.time, self.temp)
        blue = self.func_poblaciones('blue', self.time, self.temp)
        red = blue*x
        
        self.archivos()

        print('Guardando resultados')

        np.save(os.path.join(self.paths['poblaciones'],'carrier_'+self.archivos_nombres['numpy_pob']), carrier)
        np.save(os.path.join(self.paths['poblaciones'],'blue_'+self.archivos_nombres['numpy_pob']), blue)
        np.save(os.path.join(self.paths['poblaciones'],'red_'+self.archivos_nombres['numpy_pob']), red)

        poblaciones = {'carrier':carrier,
                       'blue': blue,
                       'red':red
                      }

        self.poblaciones = poblaciones
        self.p = self.poblaciones


########################################################################################################################################


    def decoherencia(self, tasa_decoherencia):
        """
        Suponiendo que los datos de la población llegan como poblacion(time,Temp),
        esta función agrega los efectos de la decoherencia a dicha población, dados por el parámetro "gamma".
        """

        self.reset()
        self.gamma = float(tasa_decoherencia)
        self.archivos()

        poblaciones_deco = {}

        x = np.exp(-self.gamma*self.time)
        termino_constante = (1-np.outer(x,np.ones(len(self.temp))))/2 
        
        print("Incluyendo decoherencia")
        for canal in self.poblaciones.keys():
            pob = termino_constante + (self.poblaciones[canal].T*x).T
            poblaciones_deco = {**poblaciones_deco, **{canal:pob}}
        

        self.poblaciones = poblaciones_deco
            


########################################################################################################################################


    def calcular_info(self, data):
        """
        Calcula de la derivada y la información de Fisher para una población dada en "data".
        """
        n = np.shape(data)[0]
        derivada = np.diff(data)/np.outer(np.ones(n),np.diff(self.temp))
        informacion = np.divide((derivada*np.outer(np.ones(n),self.temp[1:]))**2, np.multiply(data[:,1:], 1-data[:,1:]))
    
        return informacion


    def picos(self, info):
        '''
        Función que encuentra los máximos de la información de Fisher normalizada para cada temperatura. A partir de dichos máximos,
        esta función encuentra los tiempos de información máxima. Como resultado devuelve los ínidices de dichos tiempos
        para el vector 'self.time'.
        '''

        info[~np.isfinite(info)] = 0
        t_index = [np.argmax(info[:,T]) for T in range(np.shape(info)[1])]
        
        return t_index


########################################################################################################################################

    def ancho_campana(self, drop_canal=[]):
        '''
        Función que calcula el ancho medio de todas las campanas de información. Esta función se utiliza en 'self.selector_camapana' 
        para el proceso de selección. En el caso de querer desacartar un canal, el parámetro 'drop_canal' permite el ingreso de
        una lista con los nombres de los canales que se dejaran afuera del cálculo ('carrier', 'blue' o 'red').
        '''

        import warnings
        warnings.filterwarnings('ignore')

        # Calcular el ancho medio de cada camapana de información, definida en base a los tiempos máximos
        def ancho_medio(poblacion, info):
            t_index = self.picos(info)
            t_index = list(set(t_index))

            info[~np.isfinite(info)] = 0
            info_max, T_indexes = [], []

            for i,t in enumerate(t_index):
                curve = info[t,:]     
                T_index_max = np.argwhere(curve == np.nanmax(curve)).flatten()
                idx = np.argwhere(np.diff(np.sign(curve - curve[T_index_max]/2))).flatten()

                # Si solo tengo un punto que cruza la mitad de la camapana
                if len(idx) == 1:
                    index = min(idx)
                    if index < np.argmax(curve):
                        T_min = index
                        T_max = np.shape(info)[1]
                    else:
                        T_min = 0
                        T_max = index 

                # Si tengo dos índices bien definidos
                elif len(idx) == 2:
                    T_min = min(idx)
                    T_max = max(idx)

                # Si tengo más de dos indices, elijo los dos más cercanos al máximo.
                else:
                    index = np.argmax(curve)
                    inf_index = [i for i in idx if i<index]
                    sup_index = [i for i in idx if i>index]
                    T_min = max(inf_index)
                    T_max = min(sup_index)

                # Guardo los índices y el máximo de información
                T_indexes.append([int(T_min), int(T_max)])
                info_max.append(curve[int(T_index_max)])

            data = pd.DataFrame({'t_index':t_index,
                                'T_min':[T_indexes[k][0] for k in range(len(T_indexes))],
                                'T_max':[T_indexes[k][1] for k in range(len(T_indexes))],
                                'info_max':info_max
                                })

            
            return data

        if hasattr(self, 'infos') == False:
            self.cargar_infos()
            print('Carga completa')

        datos = []

        print("Detectando campanas de información")

        for key in self.infos.keys():
            if key not in drop_canal:
                data = ancho_medio(self.poblaciones[key],self.infos[key])
                data['canal'] = key
                datos.append(data)

        data = pd.concat(datos, axis=0)

        return data



    def correccion_intervalos(self, data):
        '''
        Función que corrige los intervalos de uso dados por 'self.ancho_medio()' para que las poblaciones asociadas a cada 
        intervalo sean funciónes inversibles. Esta función se utiliza en la selección dada por 'self.selector_campana()' 
        '''
        new_T_min, new_T_max = [], []
        print("Corrigiendo intervalos de uso")
        for i in range(len(data)):
            d = data.iloc[i]
            canal, t_index, T_min, T_max = d['canal'], d['t_index'], d['T_min'], d['T_max']
            p_curve = self.poblaciones[canal][t_index, :]
            sgn = np.sign(p_curve[0]-1/2)
            T_range = [ind for ind in range(T_min, T_max+1)]
            new_index = np.argwhere((sgn*(p_curve-p_curve[-1]) < 0)).reshape(1,-1)[0]
            new_T_range = np.setdiff1d(T_range,new_index)

            new_T_min.append(int(min(new_T_range)))
            new_T_max.append(int(max(new_T_range)))
        
        df = data[['t_index', 'info_max','canal']]
        df['T_min'] = new_T_min
        df['T_max'] = new_T_max
        return df



    def selector_campana(self, tol_opt=False, tolerancia=0.1, drop_canal=['blue'], tol_max=0.5, N_max=8, divisiones=50, stop=True):

        '''
        Función que realiza la selección de campanas de información. Como la selección de campanas depende de la tolerancia admitida
        para la intersección entre intervalos de uso, esta función necesita del ingreso de los siguientes parámetros para inicializarse

            - tol_opt (bool):
                - False: se seleccionará en base a la tolerancia ingresada en el parámetro 'tolerancia'.
                - True: se iniciará la búsqueda de la tolerancia óptima
            - tolerancia (float): en el caso de que se quiera seleccionar en base a una tolerancia arbitraría
            - drop_canal (list): canales que se excluiran en la selección
            - tol_max (0-1): valor máximo de la tolerancia en la búsqueda del valor óptimo
            - N_max (int): cantidad máxima de campanas a seleccionar en la búsqueda del valor óptimo
            - divisiones (int): cantidad de tolerancias a calcular entre 0 y 'tol_max'.
            - stop (bool): en el caso de que, mientras se está buscando la tolerancia óptima, se supere la cantidad 'N_max' de
                campanas máxima, este parámetro permite frenar la búsqueda (valor True).

        Como resultado, esta función almacena las campanas seleccionadas en un DataFrame de pandas en el atriburo 'self.bells'.
            
        '''

        #################################################################
        def selector(data, tol):
            data = data.sort_values('T_max')
            p = np.zeros((data.shape[0]))
            for j in range(len(p)):
                if j == 0:
                    pass
                else:
                    T_i = self.temp[data['T_min'].iloc[j]]
                    T_f = self.temp[data['T_max'].iloc[j]]
                    D = np.log10(T_f/T_i)
                    for k in reversed(range(j)):
                        t_i = self.temp[data['T_min'].iloc[k]]
                        t_f = self.temp[data['T_max'].iloc[k]]
                        d = np.log10(t_f/t_i)
                        if np.log10(t_f/T_i) < tol*np.sqrt(d*D):
                            p[j] = k
                            break

            OPT = np.zeros((data.shape[0]))

            for j in range(len(OPT)):
                if j == 0:
                    pass
                else:
                    v = data['info_max'].iloc[j]
                    OPT[j] = max([v + OPT[int(p[j])], OPT[j-1]])

            def bells():
                solution = []

                def find_solution(j):
                    if j == 0:
                        pass
                    else:
                        v = data['info_max'].iloc[j]
                        if v + OPT[int(p[j])] > OPT[j-1]:
                            solution.append(j)
                            find_solution(int(p[j]))
                        else:
                            find_solution(j-1)

                find_solution(len(OPT)-1)

                return data.iloc[solution].sort_values('T_min')

            return bells()

        #################################################################

        data = self.ancho_campana(drop_canal=drop_canal)
        data = self.correccion_intervalos(data)

        if tol_opt == False:
            try:
                print("Seleccionando los intervalos de uso para la tolerancia ingresada")
                self.bells = selector(data, float(tolerancia))
            except TypeError:
                print("Se debe ingresar un valor numérico para la tolerancia. Se recomienda ingresar un valor en 0 y 1.")

        elif tol_opt ==  True:
            tol_values = np.linspace(0,tol_max,divisiones)
            N, F_mean = [], []
            index_max = divisiones
            for i in tRange(len(tol_values), desc="Calculando tolerancia óptima"):
                if i > index_max:
                    continue
                tol = tol_values[i]
                df = selector(data, tol)
                if len(df) > N_max and stop == True:
                    index_max = i
                    continue
                N.append(len(df))
                T = []
                for j in range(len(df)):
                    T += [ind for ind in range(df['T_min'].iloc[j], df['T_max'].iloc[j]+1)]
                T = list(set(T))
                F_mean.append(df['info_max'].mean()*len(T)/len(self.temp[1:]))
            
            df = pd.DataFrame({'tol':tol_values[:index_max], 'N':N, 'info_mean':F_mean})
            mask = df['N'] <= N_max
            opt = np.argmax(df[mask]['info_mean'])

            df['opt'] = df['tol']==tol_values[opt]

            print("Guardando resultados para el valor óptimo encontrado.")
            
            self.bells = selector(data, tol_values[opt])
            self.tolerancia_opt= df


        else:
            print("Debe ingresarse un valor para tol_opt.")
            



########################################################################################################################################

    def fit(self):

        '''
        Función que realiza los ajustes de las poblaciones seleccionadas por 'self.selector_campana()'. Como resultado devuelve 
        los parámetros de las curvas sigmoideas que ajustan las poblaciones en un DataFrame, tomando de base el DataFrame almacenado
        en 'self.bells'. Al finalizar el ajuste dispara la función 'self.fit_funciones()'.

        Los resultados del ajuste se almacenan en el atributo 'self.fit_info'.
        '''

        if hasattr(self, 'bells') == False:
            return print("Se deben seleccionar las campanas de información primero.")
            
        def funcion(sgn):
            def f(x, a, b, c, m, n):
                y = (1+sgn)/2 - sgn * a * (1+m*np.exp(-b*(x-c)))/(1+n*np.exp(-b*(x-c)))
                return y
            return f

        def valores_iniciales(index):
            bell = self.bells.iloc[index]
            canal, t_index, T_min, T_max = bell['canal'], bell['t_index'], bell['T_min'], bell['T_max']
            T_range = self.temp[range(T_min, T_max+1)]

            p_curve = self.poblaciones[canal][t_index, :]
            f_curve = self.infos[canal][t_index, :]
            p_der = np.diff(p_curve)/np.diff(np.log(self.temp))

            if p_curve[0] < 1/2:
                sgn = -1
            elif p_curve[0] > 1/2:
                sgn = 1
            else:
                raise ValueError("Error. No se puede calcular el signo de la curva sigmoidea). El valor para definir es igual a '%s'"%(str(p_curve[0])))

            p_pos, p_neg = 1/2, sgn

            c_index = np.argmax(f_curve)
            c = np.log(self.temp[c_index])
            a = 1/2 - sgn*np.sqrt(1 - 4*p_pos*(1-p_pos))
            n = (1-2*p_curve[c_index]+sgn*(1-2*a)) / (2*(p_pos-p_neg)-(1-2*p_curve[c_index])-sgn*(1+2*a))
            m = ((p_pos-p_neg)/(sgn*a) - 1)*n
            b = ((1+n)**2*p_der[c_index])/(sgn*a*(m-n))

            p0 = [a,b,c,m,n]

            return sgn, p0, T_range, p_curve[range(T_min,T_max+1)]

        signo, a, b, c, m, n, p_min, p_max = [],[],[],[],[],[],[],[]

        print("Ajustando poblaciones por curvas sigmoideas")
        for i in range(len(self.bells)):
            sgn, p0, T_range, p_curve = valores_iniciales(i)
            fit = curve_fit(funcion(sgn), np.log(T_range), p_curve, p0=p0, maxfev=10000)
            signo.append(sgn), a.append(fit[0][0]), b.append(fit[0][1]), c.append(fit[0][2]), m.append(fit[0][3]), n.append(fit[0][4])

            def p(T):
                x = np.log(T)
                y = (1+sgn)/2 - sgn * fit[0][0] * (1+fit[0][3]*np.exp(-fit[0][1]*(x-fit[0][2])))/(1+fit[0][4]*np.exp(-fit[0][1]*(x-fit[0][2])))
                return y

            p_min.append(min(p(T_range))), p_max.append(max(p(T_range)))

        fit_info = pd.DataFrame({'signo':signo, 'A':a, 'beta':b, 'c':c, 'm':m, 'n':n, 'p_min':p_min, 'p_max':p_max})

        self.fit_info = pd.concat([self.bells.reset_index(drop=True), fit_info], axis=1)
        self.fit_funciones()

    


    def fit_funciones(self):

        '''
        Función que calcula, en base a los ajustes de la función 'self.fit()':
            - Los estimadores de temperatura (self.T_est) como una función
            - Las poblaciones seleccionadas (self.P_est)
            - Las campanas de información (self.F_est)

        En base a estos datos es posible la estimación de temperatura.
        '''

        P, F = [], []

        for index in range(len(self.fit_info)):

            t_index, canal = self.fit_info[['t_index','canal']].iloc[index]

            p = self.poblaciones[canal][t_index,:]
            f_curve = self.infos[canal][t_index, :]
            
            P.append(p), F.append(f_curve)

        def T_fit(pob,index):
            sgn, a, b, c, m, n = self.fit_info[['signo','A','beta','c','m','n']].iloc[index]
            q = 1-2*pob
            y = np.exp(c) * ((2*a*m - n*(1+sgn*q))/(1-2*a + sgn*q))**(1/b)
            return y

        self.T_est = T_fit
        self.P_est = np.asarray(P)
        self.F_est = np.asarray(F)
    

    def tiempos_de_medicion(self, index=None):
        '''
        Función que devuelve un DataFrame que contiene los tiempos de medición. A partir de esta función
        se le puede proveer al usuario del termómetro los tiempos de medición. Además se agregan los rangos
        de temperatura asociados para cada tiempo.

        El parámetro 'index' permite ingresar el índice de cada campana de información seleccionada dado el orden
        en el que aparecen dentro de 'self.fit_info'.
        '''

        if index==None:
            index = self.fit_info.index

        t_index = self.fit_info['t_index'].iloc[index]
        T_range = self.fit_info[['T_min','T_max']].applymap(lambda x: '%.2e'%(self.temp[x])).agg(' - '.join, axis=1).iloc[index]

        tiempos = self.transf_tiempo(self.time[t_index])
        canales = self.fit_info['canal'].apply(lambda x: x.capitalize()).iloc[index]

        return pd.DataFrame({'Tiempos de medición':tiempos, 
                             'Rango de temperatura asociado (K)':T_range,
                             'Canal de interacción': canales})


    def simular_poblacion(self, index, T, N):
        '''
        Simulación de una medición de población. Se debe ingresar:
        
            - index: índice del tiempo de medición contenido en 'self.fit_info'.
            - T: temperatura del sistema
            - N: cantidad de mediciones realizadas

        Devuelve la probabilidad simulada (float). 
        '''

        t_index, canal = self.fit_info[['t_index', 'canal']].iloc[index]
        T_index = np.argwhere(np.diff(np.sign(self.temp-T))).flatten()[0]
        prob = self.poblaciones[canal][t_index, T_index]
        p = np.random.binomial(N,prob)/N
        return p



########################################################################################################################################


    def rango_temperatura(self, T_min, T_max):
        '''
        Esta función permite calcular cuáles son los tiempos de medición que deberán utilizarse si se conoce aproximadamente
        en que rango se encuentra la temperatura del sistema. Se debe ingresar

            - T_min: temperatura mínima aproximada
            - T_max: temperatura máxima aproximada

        Dado el conocimiento previo que se tenga de sobre la temperatura del sistema, esta función devolverá los tiempos de medición
        para el intervalo [T_min, T_max].
        '''

        ind = []
        def interseccion(a,b):
            return max(0, min(a[1],b[1]) - max(a[0],b[0]))

        for i in range(len(self.fit_info)):
            inter = interseccion([T_min,T_max], [self.temp[self.fit_info['T_min'].iloc[i]], self.temp[self.fit_info['T_max'].iloc[i]]])
            if inter != 0:
                ind.append(i)
        
        self.t_index = ind

        return self.tiempos_de_medicion(index=self.t_index)



########################################################################################################################################


    def estimador(self, medicion, inf=False): 
        '''
        Estimador de la temperatura del sistema. Se debe ingresar un DataFrame como 'medicion' que contenga las columnas

            - 'P': probabilidades medidas
            - 'N': número de mediciones realizadas

        El orden de los valores ingresados debe respetar el orden de los tiempos de medición dados por 'self.tiempos_de_medición()'.

        Esta función devuelve:
            - T_est: temperatura estimada
            - s_est: error de la medición
            - r (bool): resultado de inferencia bayesiana. Dependiendo de su valor
                - True: la estimación debe ser rechazada
                - False: la estimación es aceptada
        '''

        T_aprox, sigma = [], []
        c_index = np.zeros((len(self.fit_info)))

        for i,p in enumerate(medicion['P']):
            if self.fit_info['p_min'].iloc[i] <= p <= self.fit_info['p_max'].iloc[i]:
                T = self.T_est(p,i)
                if self.temp[self.fit_info['T_min'].iloc[i]] <= T <= self.temp[self.fit_info['T_max'].iloc[i]]:
                    index = np.argwhere(np.diff(np.sign(self.temp - T))).flatten()[0]
                    T_aprox.append(T)
                    F_T =  self.F_est[i][index]
                    N = medicion['N'].iloc[i]
                    sigma.append(T/np.sqrt(N*F_T))
                    c_index[i] = 1 / sigma[-1]**2            
        
        df = pd.DataFrame({'T':T_aprox, 'sigma':sigma})
        if len(df)==0:
            T_est, s_est = np.nan, np.nan

        elif len(df)==1:
            T_est, s_est = df[['T','sigma']].iloc[0]

        else:
            if inf==False:
                T_est = np.sum(df['T']/df['sigma']**2) / np.sum(1/df['sigma']**2)
                s_est = np.sqrt(1 / np.sum(1/df['sigma']**2))
            else:
                T_est = np.mean(df['T'])
                s_est = 0

        if any(c_index):
            c_index = c_index/np.sum(c_index)

        pos, r = self.pvalue(medicion,T_est)

        return T_est, s_est, pos, r, c_index





    def estimar(self, prob, N):
        """
        Función diseñada para facilitar la estimación de temperatura dada por 'self.estimador()'. 
        Debe ingresarse un array o lista de
            - prob: probabilidades medidas
            - N: número de mediciones para cada probabilidad

        En el caso de conocer un rango aproximado de temperaturas utilizar la función 'self.rango_temperatura()' previamente
        para limitar la cantidad de valores a ingresar.

        Devuelve:
            - T_est: temperatura estimada
            - s_est: error de la medición
            - r (bool): resultado de inferencia bayesiana. Dependiendo de su valor
                - True: la estimación debe ser rechazada
                - False: la estimación es aceptada
        """

        if hasattr(self, 't_index') == True:
            index = 0
            p, n = [], []
            for i in range(len(self.fit_info)):
                if i not in self.t_index:
                    p.append(-1), n.append(1)
                else:
                    p.append(prob[index]), n.append(N[index])
                    index +=1
        else:
            p, n = prob, N

        df = pd.DataFrame({'P':p, 'N':n})
        T_est, s_est, _, r, _ = self.estimador(df)

        return T_est, s_est, r



    def pvalue(self, medicion, T_est):
        '''
        Función que calcula, mediante inferencia bayesiana, la aceptación o rechazo de una estimación. Devuelve un valor
        booleano que indica si se debe rechazar o no la estimación.
        '''

        mask = medicion['P']>0
        index = list(map(lambda x:[x], medicion[mask].index.tolist()))
        fit = self.fit_info[mask]
        medicion = medicion[mask]
        medicion['k'] = (medicion['P']*medicion['N']).astype(int)

        T_range = range(min(fit['T_min']), max(fit['T_max'])+1,10)
        pob = self.P_est[index,T_range]

        posterior = np.prod(binom(medicion['N'][:,np.newaxis], pob).pmf(medicion['k'][:,np.newaxis]), axis=0)
        C = self.temp[1]/self.temp[0]
        posterior = posterior*(C-1)*self.temp[T_range]/simps(posterior*(C-1)*self.temp[T_range])

        try:
            T_est_index = np.argwhere(np.diff(np.sign(self.temp[T_range] - T_est))).flatten()[0]
        except IndexError:
            return np.nan, True

        if posterior[T_est_index] < max(posterior)/100:
            r = True
        else:
            r = False

        return posterior[T_est_index], r



    def estimaciones(self, N=50, inf=False, e_rel=None):
        '''
        Función para simular el proceso de medición. Los parámetros a ingresar son

            - N: número de mediciones (el mismo para todos los tiempos de medición)
            - inf (bool): si se ingresa True se supondrá infinita estadística
            - e_rel: valor del error relativo en los tiempos de medición. Si se ingresa 'None' se omitiran estos errores.

        Devuelve un DataFrame que contiene

            - T_real: temperatura real del sistema
            - T_aprox: estimación de la temperatura
            - sigma: error de la estimación
            - pvalue: posterior de inferencia bayesiana
            - r: True si la estimación debe ser rechazada
        '''

        T_range = range(int(min(self.fit_info['T_min'])), int(max(self.fit_info['T_max'])+1), 10)
        T_aprox, sigma, T_real, p_value, rechazo, color_index = [], [], [], [], [], []

        for T in T_range:
            if inf == False:
                if e_rel == None:
                    p = np.random.binomial(N,self.P_est[:,T])/N 
                else:
                    t_index = self.fit_info['t_index']
                    canal = self.fit_info['canal']
                    theta = np.random.normal(loc=self.time[t_index], scale=self.time[t_index]*e_rel, size=(N,len(t_index)))
                    p = [np.random.binomial(1,self.func_poblaciones(c, theta[:,i], self.temp[T])).mean(axis=0) for i,c in enumerate(canal)]
            else:
                if e_rel == None:
                    p = self.P_est[:,T]
                else:
                    p = [self.func_poblaciones_error(row['canal'], self.time[row['t_index']], self.temp[T], 
                        self.time[row['t_index']]*e_rel) for ind, row in self.fit_info[['canal','t_index']].iterrows()]

            T_est, s_est, pos, r, c_index = self.estimador(pd.DataFrame({'P':p, 'N':N*np.ones((len(p)))}), inf=inf)
            if math.isnan(T_est):
                continue
            T_aprox.append(T_est), sigma.append(s_est), T_real.append(self.temp[T]), 
            p_value.append(pos), rechazo.append(r), color_index.append(c_index)

        return pd.DataFrame({'T_real':T_real, 'T_aprox':T_aprox, 'sigma':sigma, 'pvalue':p_value, 'r':rechazo, 'c':color_index})



########################################################################################################################################


    def error_sistematico(self):
        '''
        Cálculo del error sistemático del método. Esta función almacena la temperatura mínima 'T_S' en el 
        atributo 'self.T_S' que determina la mínima temperatura que se puede estimar y el error relativo
        cómun a todas las temperaturas en 'self.sist_max'. A partir de este error se puede calcular
        la cota máxima para el número de mediciones.
        '''

        df = self.estimaciones(inf=True)
        error = abs(1-df['T_aprox']/df['T_real'])
        index = fp(-error)[0][0]
        T_S = df['T_real'][index]
        e_max = max(error[index:])
        if sum(error>e_max) == 0:
            T_S = min(df['T_real'])
    
        self.T_S = T_S
        self.sist_max = float(e_max)



    def error_estadistico(self):
        '''
        Cálculo del error estadístico del método. Si se ingresó un valor para el error relativo en los tiempos de medición,
        esta función incluye dichos errores en la estadística. Calculando las estimaciones para todo el rango útil del termómetro
        entre 50 y 'self.med_max' mediciones, esta función devuelve la cobertura del método en 'self.cobertura'.

        Además, suponiendo que el error relativo estadístico es proporcional a la inversa de la raiz cuadrada del número de 
        mediciones, esta función almacena la un array que representa la cantidad de mediciones necesarias para tener 
        cierto error relativo. Dicho array se encuentra en 'self.n_estadistico'.
        '''

        N = range(50,self.med_max+1,10)
        r, c, rel = [], [], []
        for k in tRange(len(N), desc='Determinando el error estadísitico'):
            df = self.estimaciones(N=N[k], e_rel=self.error_t_rel)
            mask = df['pvalue']==True
            r.append(mask.astype(int).mean())
            df = df[~mask]
            c.append(((df['T_aprox']-df['sigma'] <= df['T_real'])&(df['T_real'] <= df['T_aprox']+df['sigma'])).astype(int).mean())
            rel.append((df['sigma']/df['T_aprox']).mean())
        
        e_est = pd.DataFrame({'N':N, 'rechazados':r, 'cobertura':c, 'error_relativo':rel})
        cobertura = np.mean(c)

        def f(x,a):
            y = a/np.sqrt(x)
            return y
        
        param = curve_fit(f,np.asarray(N), e_est['error_relativo'])[0][0]

        def f(x):
            y = (param/x)**2
            return y

        self.cobertura = cobertura
        self.n_estadistico = f(self.error_relativo())




    def cota_mediciones(self, min_rel):
        '''
        Función que, en base a lo calculado en 'self.error_estadistico()', permite encontrar la cota mínima de mediciones
        para las que el error relativo es menor a 'min_rel', parámtro a ingresar. Almacena dicha cota en el atributo 
        'self.min_med'.
        '''

        index = np.argwhere(np.diff(np.sign(min_rel-self.error_relativo()))).flatten()[0]
        n = int(np.ceil(self.n_estadistico[index]))
        print('Para el error relativo %.2f deberán tomarse al menos %i mediciones'%(min_rel, n))
        print("Este valor está almacenado en el atributo 'self.min_med'")
        self.min_med = n

        if n > 1/self.sist_max**2:
            print("Esta cantidad de mediciones supera la cota superior dara por el error",
            "relativo sistemático del método de estimación:", int(1/self.sist_max**2))