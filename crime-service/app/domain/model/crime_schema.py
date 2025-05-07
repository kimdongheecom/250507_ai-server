from dataclasses import dataclass
import json
import googlemaps
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables from the correct path
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
load_dotenv(env_path)

@dataclass
class CrimeSchema:
    cctv : object
    crime : object
    geo : object
    pop : object
    context : str
    fname : str 
    id : str
    label : str

    @property
    def cctv(self) -> object:
        return self._cctv
    
    @cctv.setter
    def cctv(self, cctv):
        self._cctv = cctv
    
    @property
    def crime(self) -> object:
        return self._crime
    
    @crime.setter
    def crime(self, crime):
        self._crime = crime

    @property
    def geo(self) -> object:
        return self._geo
    
    @geo.setter
    def geo(self, geo):
        self._geo = geo
    
    @property
    def pop(self) -> object:
        return self._pop
    
    @pop.setter
    def pop(self, pop):
        self._pop = pop
    
    @property
    def context(self) -> str:
        return self._context
    
    @context.setter
    def context(self, context):
        self._context = context
    
    @property
    def fname(self) -> str:
        return self._fname
    
    @fname.setter
    def fname(self, fname):
        self._fname = fname
    
    @property
    def id(self) -> str:
        return self._id
    
    @id.setter
    def id(self, id):
        self._id = id
    
    @property
    def label(self) -> str:
        return self._label
    
    @label.setter
    def label(self, label):
        self._label = label

    def new_file(self)->str:
        return self._context + self._fname

    def csv_to_dframe(self) -> object:
        file = self.new_file()
        return pd.read_csv(file, encoding='UTF-8', thousands=',')

    def xls_to_dframe(self, header, usecols)-> object:
        file = self.new_file()
        return pd.read_excel(file, encoding='UTF-8', header=header, usecols=usecols)

    def json_load(self):
        file = self.new_file()
        return json.load(open(file, encoding='UTF-8'))

    def create_gmaps(self):
        api_key = os.getenv('GOOGLE_MAPS_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_MAPS_API_KEY environment variable is not set")
        if api_key.strip() == "":
            raise ValueError("GOOGLE_MAPS_API_KEY is empty")
        return googlemaps.Client(key=api_key)
 