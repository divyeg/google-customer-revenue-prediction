import numpy as np


def largestNumber(nums):
    divfactors = []
    for i, v in enumerate(nums):
        if len(str(v)) == 1:
            divf = 10
        else:
            divf = np.power(10, (len(str(v)) - 1))
        divfactors.append(divf)
    print(divfactors)

    for i in range(len(nums) - 1):
        if len(str(nums[i])) == 1 and len(str(nums[i + 1])) != 1:
            if nums[i] % divfactors[i] < nums[i + 1] // divfactors[i + 1]:
                nums[i], nums[i + 1] = nums[i + 1], nums[i]
            if nums[i] % divfactors[i] < nums[i + 1] % divfactors[i + 1]:
                nums[i], nums[i + 1] = nums[i + 1], nums[i]
        elif len(str(nums[i])) != 1 and len(str(nums[i + 1])) == 1:
            if nums[i] // divfactors[i] < nums[i + 1] % divfactors[i + 1]:
                nums[i], nums[i + 1] = nums[i + 1], nums[i]
            if nums[i] % divfactors[i] < nums[i + 1] % divfactors[i + 1]:
                nums[i], nums[i + 1] = nums[i + 1], nums[i]
        elif len(str(nums[i])) != 1 and len(str(nums[i + 1])) != 1:
            if nums[i] // divfactors[i] < nums[i + 1] // divfactors[i + 1]:
                nums[i], nums[i + 1] = nums[i + 1], nums[i]
            if (
                nums[i] % divfactors[i] < nums[i + 1] % divfactors[i]
                and divfactors[i] <= divfactors[i + 1]
            ):
                nums[i], nums[i + 1] = nums[i + 1], nums[i]
        else:
            if nums[i] % divfactors[i] < nums[i + 1] % divfactors[i + 1]:
                nums[i], nums[i + 1] = nums[i + 1], nums[i]

    return nums


def larger_Number(nums):
    return (
        "".join(sorted([str(num) for num in nums], key=lambda x: x * 10, reverse=True))
        if set(nums) != set([0])
        else "0"
    )


nums = [3, 30, 34, 5, 9, 210]
print(larger_Number(nums))
# for i in range(len(nums)):
#     nums = largestNumber(nums)

print(nums)
