var_arr = [1,7,2,89,3]
left_pointer = var_arr[0]

print("Before: ", left_pointer)
for i in range(1, len(var_arr)):
    right_pointer = var_arr[i]
    if right_pointer > left_pointer:
        left_pointer = right_pointer
        print("Mid :", left_pointer)

print("After: ",left_pointer)
    