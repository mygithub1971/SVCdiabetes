def square(x):
    return x*x

a = [1,2,3]

# map just maps a fuction to every element in a list
asquare = list(map(square,a))
print(asquare)

asquare2 = list(map(lambda x: x*x, a))
print(asquare2)

# lambda is a function handle, you don't even need a function name
print(list(filter(lambda x: x%2, a)))



