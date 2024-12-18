
<h1  align="center">QuickInsight
</h1>
  

<h4  align="center">Ассистент для работы с научной литературой</h4>

  
  

<p  align="center">

• <a  href="#о-проекте">О проекте</a> 

• <a  href="#цель-проекта">Цель проекта</a>

• <a  href="#актуальность">Актуальность</a>

• <a  href="#анализ-области">Анализ области</a>

• <a  href="#ключевые-особенности">Ключевые особенности</a>

• <a  href="#план-реализации">План реализации</a>

• <a  href="#технологическая-основа">Технологическая основа</a>

• <a  href="#интерфейс-проекта">Интерфейс проекта</a>

• <a  href="#использование">Использование</a>

• <a  href="#доступ">Доступ</a>

• <a  href="#установка-и-запуск">Установка и запуск</a>

• <a  href="#перспективы">Перспективы</a>

• <a  href="#обратная-связь">Обратная связь</a>

</p>

  
## Структура файлов

Изображения и шрифты для сайта:

<details>

  <summary>static</summary>

- Frame 5.jpg
- gilroy-bold.ttf
- gilroy-lightitalic.ttf
- gilroy-medium.ttf
- gilroy-semibold.ttf
- gilroy-ultralight.ttf
- image 1.png
- russianrail-g-pro.otf
- style.css

</details>

Сайт:

<details>
  
<summary>templates</summary>

- index.html

</details>

Загружанные статьи(использовались для сравнения резюмирования)

<details>

  <summary>uploads</summary>

- algoritmy_obnaruzheniya_kolliziy_ploskih_dvumernyh_obektov_proizvolnoy.pdf
- Hugging_Face_Models.pdf
- lecture_1.pdf
- Lektsia_4_1.pdf
- paper88.pdf

</details>

Вспомогательные функции:

<details>

  <summary>utils</summary>

- file_to_text.py
- llm.py
- search_article.py

</details>

**app.py** -  основной код для запуска сайта

**app.ipynb** - код для установки и запуска сайта

## О проекте

  

В современном мире рабочим в научной сфере приходится тщательно изучать и анализировать огромное кол-во письменных научных работ, что отнимает очень много сил и времени.

В таких условиях AI-ассистент способный сжать, обработать и поставить главные тезисы текста— просто необходим.

  

## Цель проекта

  

Разработать научного ассистента, который поможет научным сотрудникам более эффективно управлять информацией и сократить время на обработку и анализ научных публикаций

## Актуальность
Актуальность создания AI-ассистента для работы с научной литературой обусловлена увеличением объема публикаций и научных исследований, происходящим на фоне быстрого развития науки и технологий.

Современные исследователи сталкиваются с необходимостью анализа огромного количества данных, что требует значительных временных затрат и концентрации внимания.

|  |  |
| :---: | :---: | 
| ![image](https://github.com/user-attachments/assets/776e504a-037f-4d89-8839-f5a63d28119a) | ![image](https://github.com/user-attachments/assets/2030fefb-417a-48df-9e5a-72c0c76cd28e) |



## Анализ области

  

### Процесс работы с научной литературой

  

1.  **Поиск релевантных источников**: учёные используют базы данных для поиска статей по ключевым словам или темам.

2.  **Отбор статей для изучения**: учёные отбирают статьи на основе их заголовков, аннотаций и ключевых слов, чтобы определить самое важное для их исследования.

3.  **Чтение и анализ**: дальше специалисты читают полный текст, анализируют методику, результаты и выводы.

4.  **Структурирование информации**: распределение статьи для более быстрого обращения к ней в будущем.

5.  **Цитирование и оформление ссылок**: учёные используют библиографические менеджеры для автоматизации создания ссылок и цитат в их публикациях.

  

### Проблемы при работе с научной литературой

  

-  **Объем информации** : учёные сталкиваются с проблемой “информационной перегрузки”

-  **Неполные данные или плохая структура**: иногда статьи могут быть плохо структурированы.

-  **Неэффективные методы поиска**: трудоёмкость в поиске нужных статей

-  **Время на анализ**: исследователи тратят огромное количество времени на чтение и анализ полного текста статей, даже если некоторые из них могут не быть напрямую полезными.

-  **Дублирование исследований**: существует риск дублирования исследований или недостатка знаний о существующих решениях.

## Исследовательский анализ
Большинство научных статей написаны на английском языке, он стал основным языком для научных исследований.

***Основные тематики научных статей:***

1) Технические и инженерные науки

2) Медицина и биология

3) Химия и физика

4) Компьютерные науки и IT

5) Социальные и гуманитарные науки

***Проблемы связанные с научными данными:***

1) Языковой барьер

2) Доступность научных статей

3) Релевантность и качество данных

4) Проблемы с воспроизводимостью

5) Объем и фильтрация


Эти проблемы и стимулируют создание научных ассистентов для суммаризации данных.

### Анализ конкурентов

| Scholarcy | LexRank | SummarizeBot |
| :---: | :---: | :---: |
| это веб-приложение, которое автоматически создает краткие резюме научных статей и отчетов. | это алгоритм (написанный библиотеки для Pythone “sumy”) на основе графов для автоматической суммаризации текстов, который активно используется. | универсальный инструмент, который поддерживает не только суммаризацию научных текстов, но и новостных статей, технической документации и других типов данных. |

  
  

Все предоставленные решения ( за исключением LexRank ), являются проектами, работающими на облачном хранилище.

Все решения ограничиваются лишь на сжатие текста и постановки его главных задач и смыслов.

Наше же решение способно отвечать на поставленные пользователем вопросам по данному тексту, что позволяет использовать наш ассистент в помощи при подготовки условного теоретического материала для лекций.

## API vs использование локально  

| | Локальное использование| Использование API|
| :---: | :---: | :---: |
| <p style:green>+</p> | <br>- Конфиденциальность и безопасность данных <br><br>- Независимость<br><br> - Настройка под конкретные задачи<br>(дообучение) | <br>- Быстрая интеграция и простота развертывания <br><br>- Масштабируемость<br><br> - Постоянные обновления |
| <p style:red >-</p> | <br>- Стоимость оборудования для локального запуска <br><br>- Необходимость обслуживания и управления<br><br> - Ограниченная масштабируемость | <br>- Риски для конфиденциальности данных <br><br>-Ограниченные возможности для настройки<br><br> - Задержки и зависимость от сети |

## Ключевые особенности

  

Наше решение может помочь научным специалистам не только по отдельным определённым факторам их трудоёмкой работы, а помогаем во всех поставленных нами проблемами в их сфере.

  

-  *Поиск* — наш ассистент упрощает процесс поиска определённых статей по ключевым словам и главной теме

  

-  *Обработка* — ассистент способен сжимать и суммаризировать текст. Ассистент обрабатывает исходный текст, определяет его формат (PDF, TXT, и др.), при языковой модели сжимает его и выводит тезисы с вопросами по получившемуся тексту.

  

-  *Ответы на вопросы* — ассистент способен ответить на любой вопрос по тексту.

  

## План реализации

<center><img  src="https://github.com/user-attachments/assets/44ad26ca-abf1-427e-b09a-1954ada24381" ></center>
  

## Технологическая основа

### Инструменты и билиотеки

***Python*** был выбран в качестве основного языка программирования, поскольку он является предпочтительным при работе с нейронными сетями благодаря своей простоте использования и многофункциональности.

Так же при реализации интефейса использовался язык гипертекстовой разметки  ***HTML*** , так же использовался  ***JavaScript*** , ***CSS***

### Библиотеки

  

-  ***Langchain API*** - библиотека, предоставляющая удобный инструменты для локального запуска больших моделей и обработке больших объëмов данных.

  

-  ***Transformers*** - Библиотека от компании Hugging Face, которая предоставляет удобные инструменты для загрузки и использования разнообразного количества моделей.

  

-  ***Flask*** - это простой в использовании фреймворк для создания веб-приложений на языке программирования Python.


-  ***arxiv*** - это простая в использовании библиотека для поиска и получения данных из электронного архива [arXiv.org](https://arxiv.org/).


  > Также в качестве инструмента для поиска статей по более разнообразным темам можно использовать [openalex](https://docs.openalex.org/) , библиотека **arxiv** работает заметно быстрее, но имеет ограничения по доступным темам


### Модель

В качестве основной тестовой модели использовалась ***[Pixtral 12B](https://mistral.ai/news/pixtral-12b/)*** .

Pixtral - это языковая модель от компании [Mistral](https://mistral.ai/) с поддержкой изображений.

#### Основные характеристики 
- **Общее количество параметров**: 12 миллиардов. 
-  **Архитектура**:
	  - **Vision Encoder**: 400 миллионов параметров. 
	  - **Multimodal Transformer Decoder**: комбинирует текст и изображения. 
#### Контекст и токены 
- **Максимальная длина контекста**: 128,000 токенов. 
### Поддержка языков 
- Pixtral 12B поддерживает более 20 языков. 
### Производительность
 - Высокие результаты в задачах мультимодального знания (MathVista, ChartQA). 
 - Меньшая эффективность в текстовых задачах по сравнению с моделями, такими как Claude 3 и Gemini Flash-8B. 
### Минимальные требования к видеопамяти 
- **Минимально требуется**: 48 ГБ видеопамяти (VRAM) для локального запуска.
Модель на [Hugging Face](https://huggingface.co/mistralai/Pixtral-12B-2409)

## Сравнение моделей 

Главными метриками для анализа работы модели по суммаризации являются:

-   Text MT-Bench — метрика, оценивающая способность модели следовать инструкциям в текстовых задачах.
-   Text IF-Eval — метрика также оценивает способность модели следовать инструкциям, но в более широком контексте . Эта метрика подходит для проверки того, насколько модель справляется с такой задачей.
-   MMLU (5-shot) — измеряет общие способности модели в текстовых задачах. MMLU включает задания на понимание и обработку информации из текстов, что может частично отражать способность модели к созданию осмысленных суммаризаций.

<center><img  src="https://github.com/user-attachments/assets/64ab7c4e-1099-4ef4-b216-c9380e6736df" ></center>


#### На основе этих данных можно сделать вывод, что модель Pixtral лучше своих аналогов, а также не уступает и моделям намного больше неё.


**Мы также сравнили модели для резюмирования:** 

| Pixtral | Gpt-4o | Gemini-1.5-Flash | Qwen2-72B-Instruct |
| :---: | :---: | :---: | :---: |
| ![firefox_Pmop3sHW7A](https://github.com/user-attachments/assets/9a6ce587-a41a-47a7-9881-f5f8860eb2ad) | ![firefox_P5crL23tSk](https://github.com/user-attachments/assets/e9a75935-02ee-4426-ab2f-b1dfb93fda34) | ![firefox_FYWIvttZUO](https://github.com/user-attachments/assets/99ada765-e145-4ef8-8fe4-73c77cda400e) |![lekt](https://github.com/user-attachments/assets/398aaf53-428c-400f-bbf5-551365a4169e)
| ![firefox_qr015kucK3](https://github.com/user-attachments/assets/56f3d8bc-3c3a-4bfb-ad77-a1a49a48bf0f) | ![firefox_sQjOORZB3V](https://github.com/user-attachments/assets/faea4da0-eeea-4b6f-b980-eaf3ac6226ce) | ![FW7XKXAXU9](https://github.com/user-attachments/assets/bae4ba4d-1674-4ebf-bbed-78f1cffb8e99) |![pepe](https://github.com/user-attachments/assets/21163671-7dd8-4fb2-962c-8ba45f80648e)
| ![firefox_v5ZA9FowYd](https://github.com/user-attachments/assets/1614f2b1-bc46-4f9e-94cb-6e7af6bc0355) | ![firefox_4zdMvvsJ2P](https://github.com/user-attachments/assets/0b4b0f02-a965-4462-9ca1-df3bbb45aa55) | ![firefox_0uxb5K72xW](https://github.com/user-attachments/assets/0b4fdbe5-0c76-4b43-90bb-7c72eda61b02) |![hugg](https://github.com/user-attachments/assets/529e61fb-d001-41db-aff1-3563af8c5389)
| ![firefox_Vf0JNvayJ1](https://github.com/user-attachments/assets/809a6186-ca0c-4a21-bff9-c5bbff5db537) | ![IxRWbwmptU](https://github.com/user-attachments/assets/df8ff837-2e3a-49c0-a6ed-2469e9a3d9a1) | ![JACiZ7N9jr](https://github.com/user-attachments/assets/dbbb79dc-543b-46aa-8196-2019bb0c7ece) |![algo](https://github.com/user-attachments/assets/b64df757-da7a-4d8c-8fc7-85ee913d8249)
| ![tIkMHMkd2Y](https://github.com/user-attachments/assets/e9076237-e286-4a28-9500-3e8cd16f4617) | ![firefox_8UQVDpAh01](https://github.com/user-attachments/assets/134b2f75-f1c2-4475-b6ed-b0758c9253f4) | ![vmM95bg9Sp](https://github.com/user-attachments/assets/49cd4491-8fe2-48d1-a49f-163dd09fca7b) |![lect](https://github.com/user-attachments/assets/4ed3026b-6889-4a86-8670-17441d81e7d3)


## Интерфейс проекта

  

Сайт состоит из трех страниц.


| 1 страница | 2 страница | 
|:--:|:--:|
| ![image](https://github.com/user-attachments/assets/c4392715-fb41-4ab9-b984-364394e65892) | ![image1](https://github.com/user-attachments/assets/cb4ec00c-b9c7-4cc5-b436-2890931506db) | 
| 3 страница  |
| ![image2](https://github.com/user-attachments/assets/bbdc9049-72ef-4bf2-bcf5-69573b686fef) | ![image3](https://github.com/user-attachments/assets/0a82c906-a9a3-4ca6-97d8-d0e0c6500013) |

## Использование


[https://github.com/user-attachments/assets/5d8639d8-d6c7-4962-858b-bb7104b36853](https://github.com/user-attachments/assets/33ff8ab9-7525-443b-9ce5-bcda6daed69d
)


## Доступ

На данный момент сайт работает. Его можно опробовать по ссылкам:
  - https://sirius-ai.ru/
  - https://belemort.pythonanywhere.com/

## Установка и запуск

  

Чтобы запустить локальную копию, выполните следующие простые шаги.

Учтите Python должен быть не менее **3.10.5**

```shell or cmd

# Клонируем репозиторий

> git clone https://github.com/pocketgodru/sirius_ai_biocad.git

  

# Перемещаемся в него

> cd sirius_ai_biocad
  


#Устанавливаем список библиотек

> pip install -r requirements.txt


> python app.py 

```


  

## Перспективы

**1. Увеличение скорости работы** и **улучшение качества ответов**.

**2. Добавление новых функций** по типу отдельных чатов для каждого документа .

**3. Создание полноценного сервиса**, с регистрацией пользователя и возможности выбора модели для обработки.

**4. Внедрение ещё больших функций** , по типу создания лекций    

## Обратная связь

kv.chekurin@gmail.com
