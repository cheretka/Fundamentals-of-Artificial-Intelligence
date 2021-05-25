
input_value = 2.
weight = 0.5
expected_output = 0.8
alpha = 0.1
output_value = 0.

for i in range(20):
    print(i+1)
    output_value = weight * input_value
    delta = output_value - expected_output
    weight_delta = delta * input_value
    weight = weight - alpha * weight_delta

    print("output " + str(output_value))
    error = (output_value - expected_output) ** 2
    print("error " + str(error))
    print()



