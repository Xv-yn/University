# Divide and Conquer

In summary, this is breaking up a dataset into two or more pieces and
each piece is solved recursively and recombined to produce a solution.

The time complexity can be split like so:

T(n) = divide_step + combine_step + subinstances

This idea is the origin of merge sort.

Note that divide and conquer can be used for things other than sorting

```python
def count_elements(arr):
    # Base case: If array is empty, return 0
    if len(arr) == 0:
        return 0
    # Base case: If array has one element, return 1
    if len(arr) == 1:
        return 1
    
    # Step 1: Divide - Split the array into two halves
    mid = len(arr) // 2
    left_half = arr[:mid]
    right_half = arr[mid:]
    
    # Step 2: Conquer - Recursively count elements in both halves
    left_count = count_elements(left_half)
    right_count = count_elements(right_half)
    
    # Step 3: Combine - Sum up the counts from both halves
    return left_count + right_count
```

## Master Theorem






