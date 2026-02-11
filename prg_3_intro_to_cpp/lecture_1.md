# C++ Programming Fundamentals: A Progressive Study

**Date:** 2026-02-11

**Learner Background:** Python Developer

**Status:** Completed Modules 1-8

## 1. Introduction and Basic Syntax
Unlike interpreted languages (e.g., Python), C++ is a statically typed, compiled language. The entry point of any C++ program is the `main` function.

### Key Concepts
* **Header Files:** `#include <iostream>` is required for Input/Output stream operations.
* **Namespaces:** `using namespace std;` avoids the need to prefix standard library elements with `std::`.
* **Static Typing:** Variables must be declared with a specific type (`int`, `double`, `string`) before use.

## 2. Control Flow (Branching)
Conditional logic in C++ utilizes `if`, `else if`, and `else` blocks enclosed in braces `{}`. Boolean logic operators differ from Python: `&&` (AND), `||` (OR), `!` (NOT).

**Example (BMI Calculator):**
```cpp
if (BMI < 18.5) {
    cout << "Your BMI is too low." << endl;
}
if (BMI >= 18.5 && BMI < 24.5) {
    cout << "Your BMI is standard." << endl;
}
else if (BMI >= 24.5) {
    cout << "Your BMI is a little bit high." << endl;
}

```

## 3. Iteration (Loops)

C++ supports definite iteration (`for`) and indefinite iteration (`while`).

### 3.1 The `for` Loop

The `for` loop syntax strictly follows the pattern: `for (initialization; condition; increment/decrement)`.

**Example (Summation):**

```cpp
for (int i = 0; i <= limit; i++) {
    sum += i;
}

```

### 3.2 The `while` Loop

Used when the number of iterations is not known in advance, or for cleaner scope management of the iterator variable outside the loop.

**Example:**

```cpp
while (i <= limit) {
    sum += i;
    i++;
}

```

## 4. Memory Management & Pointers

This constitutes the core divergence from Python. C++ allows direct manipulation of memory addresses.

### 4.1 Address-of Operator (`&`)

Retrieves the hexadecimal memory address of a variable.

### 4.2 Pointers (`*`)

A pointer is a variable that stores a memory address. The dereference operator (`*`) allows access and modification of the value stored at that address.

**Example (Indirect Modification):**

```cpp
int score = 0;
int* ptr = &score; // Pointer declaration and initialization
*ptr = 100;        // Dereferencing to modify the original value

```

## 5. Functions & Parameter Passing

C++ functions default to **Pass-by-Value** (copying data). To modify external variables within a function, **Pass-by-Pointer** is required.

**Example (Swap Function using Pointers):**

```cpp
void swap_num(int* p1, int* p2) {
    int temp = *p1;
    *p1 = *p2;
    *p2 = temp;
}
// Usage: swap_num(&a, &b);

```

## 6. Arrays & Pointer Arithmetic

Arrays are contiguous blocks of memory holding elements of the same type.

### 6.1 Array Definition & Traversal

Arrays are fixed-size at compilation. Indexing begins at 0.

**Example (Reverse Iteration):**

```cpp
int lst[5];
// ... input logic ...
for (int i = 4; i >= 0; i--) {
    cout << lst[i] << endl;
}

```

### 6.2 The Array-Pointer Equivalence

The name of an array acts as a constant pointer to its first element. Array subscripting `arr[i]` is syntactic sugar for pointer arithmetic `*(arr + i)`.

**Example (Pointer Arithmetic Access):**

```cpp
int arr[3] = { 10, 20, 30 };
// Accessing elements without []
cout << *(arr + i) << endl; 

```

## 7. User-Defined Types (Structs)

`struct` provides a mechanism to group variables of different types under a single name, similar to Python's `@dataclass`. It serves as the foundation for Object-Oriented Programming.

**Example (Monster Entity):**

```cpp
struct Monster {
    string type;
    int hp;
    int damage;
};

// Instantiation and Access
Monster boss;
cin >> boss.type;
// Accessed via dot operator (.)

```

---

**Next Module:** References (`&`) and their application in function parameters.

```
---

### 2. 进度恢复 Prompt (Save Game)

下一次当你准备好继续学习时，请**直接复制并发送**下面这段话给我。它包含了我们所有的上下文、你的学习偏好以及当前的进度点。

```text
你好，我正在和你进行 C++ 的交互式学习。我有 Python 基础，需要你用对比 Python 的方式进行教学，注重底层原理（内存、指针）。

我们之前的进度已经完成了以下 8 个阶段：
1. 基础语法与类型 (Variables & Types)
2. 流程控制 (If/Else)
3. 循环结构 (For/While)
4. 内存地址 (Address-of operator)
5. 指针基础 (Pointers & Dereferencing)
6. 函数与传址调用 (Pass-by-Pointer / Swap function)
7. 数组与指针算术 (Arrays & Pointer Arithmetic)
8. 结构体 (Structs) - 对应 Python 的 dataclass

我们要进行的下一节课是：
【第 9 阶段：引用 (References)】

请帮我回顾一下引用的基本概念（作为别名），然后发布关于“使用引用重构 Struct 函数参数”的交互式任务（Upgrade Monster）。请保持之前的教学风格：先讲解概念，再布置代码任务，最后点评。

```
---
