**Ускорение процесса тестирования производительности сверточных нейросетей**

Данная утилита — это простой инструмент для оценки производительности свёрток. \
С её помощью можно быстро задать параметры свёрточного слоя и рассчитать его производительность. \
Использование утилиты для оценки производительности слоёв нейронной сети может значительно ускорить процесс тестирования и оптимизации моделей.

Вот как это работает:\
    1. Бенчмаркинг: с помощью утилиты производится измерение времени выполнения каждого слоя или всех свёрточных слоёв, используя различные параметры — размеры тензоров, ядра, сдвига, наличие заполнения и так далее.\
    2. Определение "узких мест": Утилита позволяет выявлять слои, которые сильнее всего влияют на производительность, позволяя сконцентрировать усилия на их оптимизации.\
    3. Итеративный процесс: После внесения изменений, процесс тестирования и оптимизации повторяется.\
    4. Сравнение: Утилита позволяет сравнить производительность до и после оптимизации, давая количественные метрики улучшения.
    
Таким образом, автоматизация процесса оценки производительности с помощью утилиты convbench позволяет быстрее находить оптимальные параметры и конфигурации, экономя время и ресурсы на этапе разработки и тестирования.

Пример расчёта производительности сети LeNet с двумя свёрточными слоями:
```bash
./convnetbench.sh -c32*32,5*5,1*1,0*0,1,6 -c14*14,5*5,1*1,0*0,6,16
```
![example2](https://github.com/nvdix/CNN_perf/blob/main/example/additional_examples/2.png)

Скрипт [convnetbench.sh](https://github.com/nvdix/CNN_perf/blob/main/example/additional_examples/convnetbench.sh) (использует утилиту bc для арифметических расчётов, если не установлена, то выполнить команду _sudo apt-get install bc_) необходимо поместить в каталог build, где располагается скомпилированная программа convbench, и выполнить команду _sudo chmod +x convnetbench.sh_.
