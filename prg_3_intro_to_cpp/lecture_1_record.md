
# My C++ Learning Journey: Code Archive
**Date:** 2026-02-11
**Author:** (Your Name)
**Summary:** A collection of C++ programs written during the interactive learning session, covering basics from variables to structs.

---

## Phase 1 & 2: Variables, Input/Output, and Control Flow
**Project:** BMI Calculator with Health Assessment
**Key Concepts:** `double` type, `cin/cout`, `if/else if` logic, logical operators (`&&`).

```cpp
#include <iostream>
#include <string>

using namespace std;

int main()
{
	// variables
	string name;
	double height;
	double weight;

	// input
	cout << "What is your name?";
	cin >> name;
	cout << "What is your height? (m)";
	cin >> height;
	cout << "What is your weight? (kg)";
	cin >> weight;

	// calculate BMI
	double BMI = weight / (height * height);

	// output
	cout << "Hello, " << name << ", your BMI is " << BMI << "." << endl;
	if (BMI < 18.5)
	{
		cout << "Your BMI is too low." << endl;
	}
	if (BMI >= 18.5 && BMI < 24.5)
	{
		cout << "Your BMI is standard." << endl;
	}
	else if (BMI >= 24.5)
	{
		cout << "Your BMI is a little bit high." << endl;
	}

	return 0;
}

```

---

## Phase 3: Loops (Iteration)

**Project:** Summation Calculator (1 to N)
**Key Concepts:** `for` loops vs `while` loops, variable initialization (`sum = 0`).

### Version A: For Loop

```cpp
#include <iostream>
#include <string>

using namespace std;

int main()
{
	// variables
	int sum = 0;
	int limit;

	// input
	cout << "please input an integer:";
	cin >> limit;

	// calculation
	for (int i = 0; i <= limit; i++)
	{
	    sum += i;
	}

	// output
	cout << "The sum is " << sum << "." << endl;
	return 0;
}

```

### Version B: While Loop

```cpp
#include <iostream>
#include <string>

using namespace std;

int main()
{
	// variables
	int sum = 0;
	int limit;
	int i = 0;

	// input
	cout << "please input an integer:";
	cin >> limit;

	// calculation
	while (i <= limit)
	{
		sum += i;
		i++;
	}

	// output
	cout << "The sum is " << sum << "." << endl;
	return 0;
}

```

---

## Phase 4: Pointers Basics

**Project:** Modifying Values via Pointers (Indirection)
**Key Concepts:** Address-of operator (`&`), Dereference operator (`*`), Pointer declaration.

```cpp
#include <iostream>
#include <string>

using namespace std;

int main()
{
	// variables
	int score = 0;
	int* ptr = &score;

	// validation
	cout << score << endl;
	cout << ptr << endl;

	// calculation
	*ptr = 100;

	// output
	cout << score << endl;
	cout << ptr << endl;

	return 0;
}

```

---

## Phase 5: Functions & Pass-by-Pointer

**Project:** The Swap Magic
**Key Concepts:** `void` return type, passing addresses to functions, modifying external variables inside a function.

```cpp
#include <iostream>
#include <string>

using namespace std;

void swap_num(int* p1, int*p2)
{
	int temp = *p1;
	*p1 = *p2;
	*p2 = temp;
}

int main()
{
	// variables
	int a = 1;
	int b = 2;

	// output1
	cout << a << "-" << b << endl;

	// calculation
	swap_num(&a, &b);

	// output2
	cout << a << "-" << b << endl;

	return 0;
}

```

---

## Phase 6: Arrays

**Project:** Reverse Array Printer
**Key Concepts:** Array declaration (`lst[5]`), Loop bounds (0 to 4), Reverse iteration logic.

```cpp
#include <iostream>
#include <string>

using namespace std;

int main()
{
	int lst[5];
	for (int i = 0; i < 5; i++)
	{
		cout << "please input the " << i + 1 << "th number: ";
		cin >> lst[i];
	}
	for (int i = 4; i >= 0; i--)
	{
		cout << lst[i] << endl;
	}
	return 0;
}

```

---

## Phase 7: Pointer Arithmetic

**Project:** Accessing Array Elements without `[]`
**Key Concepts:** Array names are pointers, `*(arr + i)` syntax.

```cpp
#include <iostream>
#include <string>

int main()
{
	int arr[3] = { 10, 20, 30 };
	int* p = arr;
	for (int i = 0; i < 3; i++)
	{
		std::cout << *(arr + i) << std::endl;
	}
}

```

---

## Phase 8: Structs (User-Defined Types)

**Project:** RPG Monster Generator
**Key Concepts:** `struct` definition, Member access operator (`.`), Grouping related data (similar to Python dataclasses).

```cpp
#include <iostream>
#include <string>

using namespace std;

struct Monster {
	string type;
	int hp;
	int damage;
};

int main() {
	Monster boss;
	cout << "Please tell me the type of the Monster:";
	cin >> boss.type;
	cout << "Please tell me the hp of the Monster:";
	cin >> boss.hp;
	cout << "Please tell me the damage of the Monster:";
	cin >> boss.damage;

	cout << "The name of the monster is " << boss.type << ", The hp of the monster is " << boss.hp << ", The damage of the monster is " << boss.damage << "." << endl;

	return 0;
}

```
