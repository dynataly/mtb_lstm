# mtb_lstm

train_w2v.py: обучение эмбеддинга Word2Vec.

train_lstm.py: обучение модели LSTM.

w2v_lstm_functions.py: вспомогательные функции.




## Данные
Данные по туберкулёзному проекту доступны на Яндекс-диске:

Генотипы: https://disk.yandex.ru/d/Fo5ya06yKmhlPA

Там каждый файл соответствует образцу и название начинается с id образца, например SAMN08166034.
Внутри - каждая строчка означает, что соответствующий бинарный признак равен 1. Т.е. в SAMN08166034_result.tsv первая строка:
Rv2956\t-62\tC\tT\tsnp - означает, что в этом образце есть мутация в гене Rv2956, точнее в его апстриме.

Мы считаем все мутации бинарными признаками.
Бывают признаки, соответствующие инделам, а ещё - событиям ген_сломан.

Фенотипы: https://disk.yandex.ru/d/kU9IPes38t0D4g

Там файлы соответствуют препаратам. В каждом файле информация об устойчивости образцов. 1 - резистентный, 0 - устойчивый.
Не ко всем образцам есть все фенотипы. 

Наша фишка - признаки - домены генов. Они лежат отдельно:

https://disk.yandex.ru/d/B9DpbpuTEmtejQ

Они для каждого препарата свои. При этом, мы разбили выборку на 5 фолдов для кросс-валиации. И на этих фолдах обучались пороги, используемые для создания этих признаков. Потому в каждой папке, соответствующей препарату, есть ещё 5 папок для фолдов.
А там уже знакомые нам бинарные признаки.
В каждой папке вся выборка. То, как разбить её на train и test можно увидеть в

https://disk.yandex.ru/d/3laxgUF7h0Y8_g

Там для каждого препарата есть файлы типа Rifampicin.5_fold_split.txt - там показано, как разбить образцы из папки fold5 на train и test.

Препринт:

https://www.biorxiv.org/content/10.1101/2022.03.16.484601v2
