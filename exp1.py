import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
from cirq import Circuit
from cirq.devices import GridQubit
from cirq import Simulator
import numpy as np 
import sympy as sp
from numpy import genfromtxt

c = genfromtxt('data.csv', delimiter=',')
# print(c.shape)
np.random.shuffle(c)
x_train=c[:,:4]
print(x_train.shape)
x_train_n = np.pi*((x_train - x_train.min(0)) / 2*x_train.ptp(0))
# x_test_n = np.pi*((x_test - x_test.min(0)) / 2*x_test.ptp(0))

x_train_n = x_train_n[:100,:]
x_test_n= x_train_n[80:100,:]

# print(x_train_n.max())
# print(x_train_n.min())
# exit()

y_train=c[:100,4]
y_test = c[80:100,4]
# y_train=y_train*2-1
# print(y_train)
# exit()
# print(y_train.shape)
# y_train_cat = tf.keras.utils.to_categorical(y_train)
# y_test = tf.keras.utils.to_categorical(y_test)
# print(y_train)
# exit()
# y_train=y_train*2-1

def convert_to_circuit(values):
	qubits = cirq.GridQubit.rect(1,4)
	circuit = cirq.Circuit()
	for i, value in enumerate(values):
		rot = cirq.ry(value*2/np.pi)
		circuit.append(rot(qubits[i]))
		# circuit.append(cirq.X(qubits[i]))

	# print(circuit)
	return circuit

x_train_cirq = [convert_to_circuit(x) for x in x_train_n]
x_test_cirq = [convert_to_circuit(x) for x in x_test_n]
# print(x_train_cirq)
x_train_tf_circ = tfq.convert_to_tensor(x_train_cirq)
x_test_tf_circ = tfq.convert_to_tensor(x_test_cirq)
# print(x_test_tf_circ)

def one_bit_unitary_new(circuit, symbols, bit):
	circuit.append(cirq.X(bit)**symbols[0])
	circuit.append(cirq.Y(bit)**symbols[1])
	circuit.append(cirq.Z(bit)**symbols[2])


def one_bit_unitary(bit, symbols):
	rot = cirq.ry(symbols)
	return rot(bit)

def create_model_new():
	data_qubit = cirq.GridQubit.rect(1,4)
	readout = cirq.GridQubit(-1,-1)
	circuit = cirq.Circuit()
	symbols = sp.symbols('x0:100')
	k=0
	for i, bit in enumerate(data_qubit):
		one_bit_unitary_new(circuit, [symbols[k], symbols[k+1], symbols[k+2]],bit)
		k=k+3
	circuit.append(cirq.CNOT(data_qubit[0], data_qubit[1]))
	circuit.append(cirq.CNOT(data_qubit[3], data_qubit[2]))
	one_bit_unitary_new(circuit, [symbols[k], symbols[k+1], symbols[k+2]],data_qubit[1])
	k=k+3
	one_bit_unitary_new(circuit, [symbols[k], symbols[k+1], symbols[k+2]],data_qubit[2])
	k=k+3
	circuit.append(cirq.CNOT(data_qubit[1], data_qubit[2]))
	one_bit_unitary_new(circuit, [symbols[k], symbols[k+1], symbols[k+2]],data_qubit[2])
	k=k+3
	print (circuit)
	return circuit, cirq.Z(data_qubit[2])
model_new, readout_new = create_model_new()
model = tf.keras.Sequential([
		tf.keras.layers.Input(shape=(), dtype=tf.string),
		tfq.layers.PQC(model_new, readout_new),
		tf.keras.layers.Activation('sigmoid')
		])
model.summary()
sgd = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.75, nesterov=True)
def hinge_accuracy(y_true, y_pred):
    y_true = tf.squeeze(y_true) > 0.0
    y_pred = tf.squeeze(y_pred) > 0.0
    result = tf.cast(y_true == y_pred, tf.float32)

    return tf.reduce_mean(result)
model.compile(loss = 'binary_crossentropy',#tf.keras.losses.Hinge(),
			  optimizer = 'adam',
			  metrics=['accuracy'])
model.fit(x_train_tf_circ, y_train,shuffle=True, batch_size = 32,epochs=100, validation_split=0.2)
scores = model.evaluate(x_train_tf_circ, y_train, verbose=0)
scores3 = model.evaluate(x_test_tf_circ, y_test, verbose=0)
print('Accuracy on training data: {} \n Error on training data: {}'.format(scores[1], 1 - scores[1]))
print('Accuracy on test data: {} \n Error on test data: {}'.format(scores3[1], 1 - scores3[1]))


model2 = tf.keras.Sequential([
  tf.keras.layers.Dense(4, activation=tf.nn.relu, input_shape=x_train_n[0].shape),
  tf.keras.layers.Dense(1, activation='sigmoid')
])
model2.summary()
model2.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])
model2.fit(x_train_n, y_train, batch_size=32, validation_split=0.2, epochs=100)
scores2 = model2.evaluate(x_train_n, y_train, verbose=0)
scores4 = model2.evaluate(x_test_n, y_test, verbose=0)
print('Classical ML: Accuracy on training data: {} \n Error on training data: {}'.format(scores2[1], 1 - scores2[1]))
print('Quantum ML: Accuracy on training data: {} \n Error on training data: {}'.format(scores[1], 1 - scores[1]))
print('Classical ML: Accuracy on test data: {} \n Error on test data: {}'.format(scores4[1], 1 - scores4[1]))
print('Quantm ML: Accuracy on test data: {} \n Error on test data: {}'.format(scores3[1], 1 - scores3[1]))
