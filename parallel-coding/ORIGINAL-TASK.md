# Лабораторная работа №0

[//]: # (Microsoft Word - Lab0.docx, 18/02/2022, 14:11:19)
а) Случайно сгенерировать файл в 10 тысяч символов, используя указанный алфавит
(заглавные и строчные буквы). 

```shell
(cd ../cmake-build-debug/L0
./lab0 generate <alphabet.txt >10k.in)
```

Закодировать текстовую информацию, используя указанный в
задании алгоритм. Определить цену кодирования. 

```shell
(cd ../cmake-build-debug/L0
./lab0 encode_huffman 10k.in >10k.out.dat)
```

```
цена кодирования = 4.7018
коэффициент сжатия = 0.6011
```

Декодировать информацию, определить коэффициент сжатия. 

```shell
(cd ../cmake-build-debug/L0
./lab0 decode_huffman <10k.out.dat >10k.decoded)
```

Программно проверить, что декодирование произошло верно.

```shell
(cd ../cmake-build-debug/L0
./lab0 encode_huffman 10k.in | ./lab0 decode_huffman | cmp 10k.in - && (echo "same!") || (echo "different!"))
```

```
цена кодирования = 4.7018
коэффициент сжатия = 0.6011
same!
```

б) Закодировать информацию алгоритмом RLE. Декодировать информацию, определить
коэффициент сжатия. Программно проверить, что декодирование произошло верно.

```shell
(cd ../cmake-build-debug/L0
./lab0 encode_rle <10k.in | ./lab0 decode_rle | cmp 10k.in - && (echo "same!") || (echo "different!"))
```

```
коэффициент сжатия = 1.959
same!
```

в) Аналогично пункту а) закодировать информацию, применив двухступенчатое кодирование
(RLE + алгоритм варианта и алгоритм варианта + RLE). Показать, какой из способов более
эффективный.
```shell
(cd ../cmake-build-debug/L0
./lab0 encode_huffman 10k.in | ./lab0 encode_rle | ./lab0 decode_rle | ./lab0 decode_huffman | cmp 10k.in - && (echo "same!") || (echo "different!"))
```

```
цена кодирования = 4.7018
коэффициент сжатия = 0.6011
коэффициент сжатия = 1.98436
same!
```

Суммарный коэффициент сжатия $0.6011 * 1.98436 = 1.193$

```shell
(cd ../cmake-build-debug/L0
./lab0 encode_rle <10k.in >10k.rle.out && ./lab0 encode_huffman 10k.rle.out | ./lab0 decode_huffman | ./lab0 decode_rle | cmp 10k.in - && (echo "same!") || (echo "different!")
rm 10k.rle.out)
```

```
коэффициент сжатия = 1.959
цена кодирования = 3.43017
коэффициент сжатия = 0.4366
same!
```

Суммарный коэффициент сжатия $3.43017 * 0.4366 = 1.498$

<details>
<summary>Варианты</summary>

- A — Алгоритм Фано
- B — Алгоритм Хаффмена
- C — Алгоритм LZW

1. {а, б, в, г, д, е, пробел, 0-9, %, .} Алгоритм A
2. {ж, з, и, к, л, м, пробел, 0-9, @, .} Алгоритм B
3. {н, о, п, р, с, т, пробел, 0-9, $, .} Алгоритм C
4. {н, о, п, р, с, т, пробел, 0-9, ^, .} Алгоритм A
5. {у, ф, х, ц, ч, ш, пробел, 0-9, &, .} Алгоритм B
6. {щ, ь, ы, ъ, э, ю, пробел, 0-9, *, .} Алгоритм C
7. {н, о, п, р, с, т, пробел, 0-9, ! , .} Алгоритм A
8. {я, о, в, п, д, т, пробел, 0-9, (, .} Алгоритм B
9. {а, ы, и, б, к, с, пробел, 0-9, ), .} Алгоритм C
10. {е, ф, б, н, ь, ю, пробел, 0-9, №, .} Алгоритм A
11. {ё, э, д, ж, р, м, пробел, 0-9, #, .} Алгоритм B
12. {к, ц, й, а, о, ч, пробел, 0-9, ?, .} Алгоритм C
13. {у, к, н, ш, э, я, пробел, 0-9, /, .} Алгоритм A
14. {с, м, и, ъ, е, т, пробел, 0-9, <, .} Алгоритм B
15. {й, ё, п, в, о, д, пробел, 0-9, >, .} Алгоритм C

</details>

## References

[Compression using Huffman Codes](https://www.cs.utexas.edu/users/djimenez/utsa/cs1723/lecture14.html)

[Hackers Delight](https://doc.lagout.org/security/Hackers%20Delight.pdf)

- http://algolist.manual.ru/compress/standard/huffman.php
- https://www.cs.utexas.edu/users/djimenez/utsa/cs1723/assign7/assign7.html
- https://www.cs.utexas.edu/users/djimenez/utsa/cs1723/

RLE

- https://github.com/ttangeman/rle/blob/master/rle.c