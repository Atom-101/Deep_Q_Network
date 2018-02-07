class Light(object):
	max_luminosity = 0
	max_power = 0

	def __init__(self,luminosity,power):
		self.max_luminosity = luminosity
		self.max_power  = power
		self.luminosity = 0
		self.power = 0

	def set_intensity(self,i):
		self.luminosity = i*self.max_luminosity
		self.power = i*self.max_power
		
	def get_intensity(self):
		return self.luminosity,self.power
		
