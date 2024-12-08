import numpy as np
import math
import matplotlib.pyplot as plt

def pw_laplacian_1d(array: np.array, left_point: int, mid_point: int, right_point: int, delta: float):
    try:
        return (array[left_point] + array[right_point] - 2 * array[mid_point]) / (delta ** 2)
    except IndexError:
        print('Index not found')      
    
def pw_heat_equation(array: np.array, left_point: int, mid_point: int, right_point: int, x_delta: float, t_delta: float):
    return pw_laplacian_1d(array, left_point, mid_point, right_point, x_delta) * t_delta

class PeriodicLattice1D:
    def __init__(self, length: float, delta: float):
        self.length = length
        self.delta = delta
        self.array_size = int(length // delta) + 1
        self.x_axis = np.linspace(0, 1, self.array_size)
        self.array = self.create_lattice()
        self.array_t_minus_1 = np.copy(self.array)
        self.mode1 = np.sin(np.pi * self.x_axis) * 2

    def create_lattice(self):
        return np.zeros(self.array_size)

    def random_initial(self):
        for i in range(0, self.array_size):
            self.array[i] = np.random.normal(0, 1)
        self.array_t_minus_1 = np.copy(self.array)

    #pointwise Gaussian random initial condition with Dirichlet boundaryy conditions
    def random_initial_dirichlet(self):
        self.dirichlet = 1
        for i in range(1, self.array_size):
            self.array[i] = np.random.normal(0, 1)
        self.array_t_minus_1 = np.copy(self.array)

    def initial_data_dirichlet(self, function: callable):
        for i in range(1, self.array_size):
            self.array[i] = function(i / self.array_size)
        self.array_t_minus_1 = np.copy(self.array)

    def initial_data(self, function: callable):
        for i in range(0, self.array_size):
            self.array[i] = function(i / self.array_size)
        self.array_t_minus_1 = np.copy(self.array)

    def random_perturbation_dirichlet(self, size: float):
        for i in range(1, self.array_size):
            self.array[i] += np.random.normal(0, size)

    def random_perturbation(self, size: float):
        for i in range(0, self.array_size):
            self.array[i] += np.random.normal(0, size)

    def graph(self):
        current_graph = plt.plot(self.x_axis, self.array)
        average = np.full(self.array_size, self.average())
        #half_mode_graph = np.sin(np.pi * self.x_axis) * self.num_int_sin_mode1()
        first_mode_graph = self.nth_mode(1)
        #second_mode_graph = self.nth_mode(2)
        plt.plot(self.x_axis, average)
        #plt.plot(self.x_axis, half_mode_graph)
        plt.plot(self.x_axis, first_mode_graph)
        #plt.plot(self.x_axis, second_mode_graph)

    def laplacian(self):
        new_array = np.zeros(len(self.array))
        for i in range(1, self.array_size - 1):
            new_array[i] = pw_laplacian_1d(self.array, i-1, i, i+1, self.delta)
        new_array[self.array_size - 1] = pw_laplacian_1d(self.array, self.array_size-2, self.array_size-1, 0, self.delta)
        return new_array

    def heat_flow_step(self, t_delta: float):
        self.array = self.array + self.laplacian() * t_delta

    def heat_flow_step_phi4(self, t_delta: float, mass: float, phi4_parameter: float):
        self.array = self.array + (self.laplacian() - mass * self.array - phi4_parameter * self.array ** 3) * t_delta 

    def wave_flow_step(self, t_delta: float, damping: float):
        copy = np.copy(self.array)
        self.array = 2 * self.array - self.array_t_minus_1 + (-damping * self.array + self.laplacian()) * (t_delta ** 2)
        self.array_t_minus_1 = copy

    def ft(self):
        self.fourier = np.fft.fft(self.array)

    def smooth(self, wavenumber_limit: int):
        self.fourier = np.fft.fft(self.array)
        self.fourier[wavenumber_limit : self.array_size - wavenumber_limit] = 0.1 * self.fourier[wavenumber_limit : self.array_size - wavenumber_limit]
        self.array = np.fft.ifft(self.fourier).real
        self.array[0] = 0

    def nth_mode(self, n: int):
        fourier = np.fft.fft(self.array)
        imag_part = np.zeros(len(fourier), dtype=complex)
        imag_part[n] = 1j
        imag_part[-n] = 1j
        fourier[0: n] = 0
        fourier[n+1: len(fourier) - n] = 0
        if n > 1:
            fourier[len(fourier) - n + 1:] = 0
        fourier = fourier.imag.astype(complex) * imag_part
        return(np.fft.ifft(fourier).real)
    
    def num_int_sin_mode1(self):
        return sum(self.mode1 * self.array) * self.delta
    
    def average(self):
        return sum(self.array) / self.array_size

    def __str__(self):
        return f'{self.array}'
        
if __name__ == "__main__":
    #0.05 for heat is good
    test_lattice = PeriodicLattice1D(1, 0.05)
    
    plt.ion
    plt.ylim(-1.5, 1.5)
    plt.xlim(0,1)
    #test_lattice.random_initial()
    while True:
        command = int(input('Command: '))
        if command == 0:
            break
        elif command == 1:
            i = 0
            test_lattice.graph()
            test_lattice.initial_data_dirichlet(lambda t: np.sin(2 * np.pi * t))
            input("Start?")
            while i < 500:
                plt.pause(0.001)
                plt.clf()
                plt.ylim(-2.5, 2.5)
                plt.xlim(0,1)
                test_lattice.heat_flow_step_phi4(0.00025, -1, 1)
                test_lattice.random_perturbation(0.1)
                test_lattice.graph()
                i += 1
                print(i)
        elif command == 2:
            i = 0
            test_lattice.initial_data_dirichlet(lambda t: np.sin(2 * np.pi * t) + np.sin(np.pi * t) )
            #test_lattice.random_initial_dirichlet()
            test_lattice.graph()
            input("Start?")
            while i < 100:
                plt.pause(0.001)
                plt.clf()
                plt.ylim(-2.5, 2.5)
                plt.xlim(0,1)
                test_lattice.wave_flow_step(0.005, 10)
                #test_lattice.random_perturbation_dirichlet(0.001)
                #test_lattice.smooth(20)
                test_lattice.graph()
                i += 1
                print(i)
        elif command == 3:
            x_axis = np.linspace(0,1,100)
            test = np.sin(np.pi * x_axis) ** 2 / 50
            print(sum(test))
        elif command == 4:
            x_axis = np.linspace(0,1,100)
            test = np.zeros(100, dtype=np.complex128)
            test[2] = 50j
            test[-2] = -50j
            tfft = np.fft.ifft(test)
            plt.plot(x_axis, tfft)
            plt.show()

            
            

    