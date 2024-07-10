# Prezicere Consum de Curent

Aceest proiect are in vedere creearea unei inteligente artificiale de tipul machine learning pentru prezicerile de consum in functie de: Data, Zona, Municipiu, Tip de folosinta si Statutul social al consumatorului, folosindu-se un model LSTM preluat de pe: https://keras.io/examples/timeseries/timeseries_weather_forecasting/. 

Datele au fost extrase din PCSTCOL: Power consumption data from an area of southern Colombia, publicat in 14-Jan-2020: https://data.mendeley.com/datasets/xbt7scz5ny/3. Au fost extrase 4427 de caracteristici din principalele 7 municipii din Nariño, Colombia, din Decembrie 2010 pana in Mai 2016. Aceeste date au fost extrase de catre CEDENAR(Centrales Eléctricas de Nariño).
Modelul parcurge 10 epoci, cu serii de 265. 

Arhitectura aleasa este de: un strat [LSTM](https://en.wikipedia.org/wiki/Long_short-term_memory)(Long short-term memory) cu 32 de neuroni si un strtat dens de iesire cu 1 neuron. 

## Cuprins

- [Instalare](#instalare)
- [Utilizare](#utilizare)
- [Licenta](#licenta)

## Instalare

Cum aceest proiect este un singur fisier poate fi descarcat/clonat dupa cum este preferat: 
```
https://github.com/SasZombie/PowerPrediction
```
Dependentele necesare: **python, keras, pandas, sklearn, numpy, matplotlib**

### Windows

Se instaleaza python de pe internet, iar apoi:

```
pip install keras, pandas, sklearn, numpy, matplotlib
```

### Linux/MacOS

Trebuie instalate aceeleasi pachete de la managerul de pachete respectiv fiecarei distribuitii.

> [!IMPORTANT]
> Ar putea exista un conflict intre keras si numpy 2.0, ceea ce face compilarea imposibila. Pentru o instalare fara probleme se recomanda venv

```
venv currentvenv
source venv/bin/activate
```
Acum aceest terminal se va comporta ca cel din windows si se procedeaza precum in [windows](#windows)

## Utilizare

Pentru rulare se poate folosii:
```
python3 main.py
```
Modelul se va antrena si va incerca sa prezica 5 noi valori. Deoarece este vorba de curent, o diferenta nu se poate observa usor intre 5 zile consecutive, dar exista si exceptii. Daca se doreste a modifica parametrii, va fi necesare o reantrenare a retelei. Dupa antrenare modelul va arata performanta sa in format tip graf.

> [!NOTE]
> Dintr-un motiv apar 2 X-uri in plus la verificare, nu stiu cum de apar sau cum sa le scot la aceest moment



## Licenta

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files, to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
