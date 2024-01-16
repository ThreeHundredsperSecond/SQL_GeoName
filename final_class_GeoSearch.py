
class GeoSearch:
    def __init__(self, database=DATABASE, table_name='search_geoname',
                 create_table=False, use_transformer=True, pretrained_model=r'C:\model',
                 countries=COUNTRY_CODES, population_size=15000,  uuid='corpus_embeddings_array'):
        '''
        Инициализация объекта GeoSearch.

        Параметры:
        - database: Параметры подключения к базе данных (по умолчанию переменная DATABASE).
        - table_name: Название таблицы (по умолчанию 'search_geoname').
        - create_table: Флаг для создания таблицы (по умолчанию False).
        - use_transformer: Флаг использования предобученной модели (по умолчанию True).
        - pretrained_model: Ссылка на предобученную модель.
        - countries: код стран для поиска 
        - population_size: минимальный порог населения
        - uiid: название записи с векторным представлением имен населенных пунктов  

        Внутренние переменные:
        - database: Параметры подключения к базе данных.
        - table_name: Название таблицы.
        - df: DataFrame для хранения данных из таблицы.
        - model: Предобученная модель для трансформера.
        - corpus_embeddings: Массив embeddings.
        - use_transformer: Флаг использования трансформера.

        Действия:
        - Установка соединения с базой данных.
        - Создание таблицы, если указан флаг create_table.
        - Чтение данных из таблицы в DataFrame.
        - Проверка наличия необходимых столбцов в таблице.
        - Инициализация предобученной модели и массива embeddings, если указан флаг use_transformer.
        '''
        # Установка соединения с БД
        self.database = database
        self.database_connection()
        # Устанавливаем значение uuid
        self.uuid = uuid

        # Название таблицы
        self.table_name = table_name
        self.countries = countries
        self.population_size = population_size

        # Если нужно создать таблицу
        if create_table:
            # Вызов метода для создания таблицы
            self.create_table()
        else:
            # SQL-запрос для выборки данных из таблицы
            self.query = f"SELECT * FROM {self.table_name}"
            # Чтение данных из БД в DataFrame
            self.df = pd.read_sql_query(self.query, con=self.engine)

        # Проверка наличия необходимых столбцов в таблице
        self.check_columns()

        # Использовать ли здесь модель
        if use_transformer:
            # Инициализация предобученной модели
            self.model = SentenceTransformer(pretrained_model)

            # Получение массива embeddings
            self.corpus_embeddings = self.get_embeddings_array()

            # Если массив embeddings пуст, сохранение embeddings
            if self.corpus_embeddings is None:
                self.corpus_embeddings = self.create_corpus_embeddings()
                self.save_embeddings(self.corpus_embeddings)

            # Объявление флага
            self.use_transformer = use_transformer
        else:
            self.use_transformer = use_transformer

    def database_connection(self):
        '''Метод для подключения к БД'''

        # Параметры подключения
        self.db_connect_kwargs = {
            'user': self.database.get('username'),
            'password': self.database.get('password'),
            'host': self.database.get('host'),
            'port': self.database.get('port'),
            'dbname': self.database.get('database')
        }

        # Создаем подключение к базе данных
        self.connection = psy.connect(**self.db_connect_kwargs)

        # Устанавливаем автокоммит
        self.connection.set_session(autocommit=True)

        # Инициализируем курсор
        self.cursor = self.connection.cursor()

        # Создаем SQLAlchemy engine для дополнительных возможностей
        self.engine = create_engine(URL(**self.database))

    def execute_sql_query(self):
        '''Метод выполняет SQL-запрос с возможностью указать страны и население'''

        # SQL-запрос с использованием параметров
        self.query = f'''
            SELECT 
                gc.geonameid,
                gc.name,
                gc.asciiname,
                gc.alternatenames,
                gc.country AS country_code,
                gc.admin1,
                gc.population,
                ci.country,
                ac.ascii_name AS region
            FROM 
                geoname gc
            LEFT JOIN admin_codes ac ON gc.admin1::TEXT = split_part(ac.concatenated_codes, '.', 2)::TEXT
                AND gc.country = split_part(ac.concatenated_codes, '.', 1)::TEXT
            LEFT JOIN country_info ci ON gc.country = ci.country_code
            WHERE 
                gc.population > {self.population_size}
                AND gc.country IN {tuple(self.countries)}
        '''

        # Выполнение SQL-запроса и возврат результата в виде DataFrame
        return pd.read_sql_query(self.query, con=self.engine)

    def create_table(self):
        '''
        Метод создает таблицу в БД, используя запрос по региону и населению.
        1. Создание DataFrame с использованием SQL-запроса.
        2. Фильтрация строк, где 'name' является NaN.
        3. Фильтрация строк, где 'asciiname' является NaN.
        4. Заполнение пропущенных значений в столбце 'region' значениями из столбца 'name'.
        5. Запись DataFrame в БД с указанием имени таблицы, индекса, и режима замены (replace, если таблица уже существует).
        '''

        # Создание DataFrame с использованием SQL-запроса
        self.df = self.execute_sql_query()

        # Фильтрация строк, где 'name' является NaN
        self.df = self.df[~self.df['name'].isna()]

        # Фильтрация строк, где 'asciiname' является NaN
        self.df = self.df[~self.df['asciiname'].isna()]

        # Заполнение пропущенных значений в столбце 'region' значениями из столбца 'name'
        self.df['region'].fillna(self.df['name'], inplace=True)

        self.df.to_sql(self.table_name, con=self.engine,
                       index=False, if_exists='replace')

    def check_columns(self):
        '''
        Метод проверяет наличие необходимых столбцов в DataFrame.
        1. Задание списка обязательных столбцов.
        2. Проверка отсутствия каждого обязательного столбца в DataFrame.
        3. Если какие-то столбцы отсутствуют, вызывается исключение ValueError с указанием отсутствующих столбцов.
        '''

        # Список обязательных столбцов
        required_columns = ['geonameid', 'name',
                            'region', 'country', 'asciiname']

        # Столбцы, которые отсутствуют в DataFrame
        missing_columns = [
            col for col in required_columns if col not in self.df.columns]

        # Если есть отсутствующие столбцы, вызвать исключение ValueError
        if missing_columns:
            missing_columns_str = ', '.join(missing_columns)
            raise ValueError(f"Отсутствующие столбцы: {missing_columns_str}. "
                             f"Необходимо наличие всех обязательных столбцов в DataFrame.")

    def get_embeddings_array(self):
        '''
        Метод выполняет SQL-запрос к БД для извлечения массива np_array_bytes из таблицы numpy_arrays, используя заданный uuid.

        Параметры:
        - uuid: Уникальный идентификатор массива (по умолчанию 'corpus_embeddings_array').

        Возвращаемое значение:
        - Восстановленный массив, если он найден; в противном случае, возвращается None.
        '''

        # Установка соединения с БД
        self.database_connection()

        try:
            # Выполняем SQL-запрос для выбора массива np_array_bytes по uuid
            self.cursor.execute(
                """
                SELECT np_array_bytes
                FROM corpus_embeddings
                WHERE uuid=%s
                """,
                (self.uuid,)
            )

            # Получаем результат
            self.result = self.cursor.fetchone()

            # Если результат существует
            if self.result:
                # Восстанавливаем массив из байтового представления
                self.retrieved_array = pickle.loads(self.result[0])

                # Закрываем соединение и курсор
                self.close_connection()

                return self.retrieved_array
            else:
                # Выводим сообщение о том, что массив с указанным uuid не найден
                print(f"Массив {self.uuid} с указанным uuid не найден.")
                return None
        except Exception as e:
            # Обработка исключения, например, вывод ошибки
            print(f"Ошибка при выполнении SQL-запроса: {e}")
            return None
        finally:
            # Закрываем соединение и курсор в блоке finally, чтобы гарантировать их закрытие
            self.close_connection()

    def close_connection(self):
        '''Метод закрывает соединение и курсор с БД.'''
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()

    def create_corpus_embeddings(self):
        '''
        Метод создает embeddings для списка имен (corpus) с использованием предобученной модели.

        Возвращаемое значение:
        - массива embeddings.
        '''

        # Получение списка имен из DataFrame
        corpus = self.df['name'].tolist()

        # Использование модели для векторизации списка имен
        embeddings = self.model.encode(
            corpus, convert_to_tensor=False)

        # Возвращение массива embeddings
        return embeddings

    def save_embeddings(self, array=None, uuid=None, drop_table=True):
        '''
        Метод сохраняет массив в базу данных в виде байтового представления.

        Параметры:
        - array: Массив для сохранения (по умолчанию self.corpus_embeddings).
        - uuid: Уникальный идентификатор массива (по умолчанию None).
        - drop_table: Флаг, определяющий, следует ли удалять таблицу перед сохранением новых данных (по умолчанию True).
        '''

        # Создание подключения к базе данных
        self.database_connection()

        # Создание таблицы corpus_embeddings (удаляется, если drop_table установлен в True)
        if drop_table:
            self.cursor.execute(
                """
                DROP TABLE IF EXISTS corpus_embeddings;
                CREATE TABLE corpus_embeddings (
                    uuid VARCHAR PRIMARY KEY,
                    np_array_bytes BYTEA
                )
                """
            )

        # Вставка массива в базу данных
        self.array = array if (array is not None) and (
            len(array) > 0) else self.corpus_embeddings

        # Использование uuid экземпляра класса, если не предоставлен через параметры метода
        self.uuid = uuid if uuid is not None else self.uuid

        self.cursor.execute(
            """
            INSERT INTO corpus_embeddings(uuid, np_array_bytes)
            VALUES (%s, %s)
            """,
            (self.uuid, pickle.dumps(self.array))
        )

        # Закрытие соединения
        self.close_connection()

    def initialize_model(self, pretrained_model, new_corpus=False, uuid=None, drop_table=True):
        '''
        Инициализация модели SentenceTransformer и массива embeddings.

        Параметры:
        - pretrained_model: Путь или название предобученной модели для SentenceTransformer.
        - new_corpus: Флаг, указывающий, следует ли использовать новый корпус (по умолчанию False).
        - uuid: Уникальный идентификатор массива embeddings (по умолчанию 'corpus_embeddings_array').
        - drop_table: Флаг, определяющий, следует ли удалять таблицу перед сохранением новых данных (по умолчанию True).
        '''

        # Инициализация предобученной модели SentenceTransformer
        self.model = SentenceTransformer(pretrained_model)

        # Получение массива embeddings, если не указано использование нового корпуса
        if not new_corpus:
            # Использование uuid экземпляра класса, если не предоставлен через параметры метода
            self.uuid = uuid if uuid is not None else self.uuid
            self.corpus_embeddings = self.get_embeddings_array()
        else:
            self.corpus_embeddings = self.create_corpus_embeddings()
            self.save_embeddings(self.corpus_embeddings,
                                 uuid=uuid, drop_table=drop_table)

        # Установка флага для использования трансформера
        self.use_transformer = True

    def get_lev_distance(self, queries_name, translator=False):
        '''
        Метод находит ближайшие города в столбце 'asciiname' с использованием расстояния Левенштейна.

        Параметры:
        - queries_name: Список запросов (названий городов) для поиска ближайших совпадений.
        - translator: Флаг использования переводчика (по умолчанию False).

        Возвращаемое значение:
        - DataFrame с информацией о ближайших совпадениях для каждого запроса.
        '''

        # Список городов из столбца 'asciiname'
        city_names = self.df['asciiname']

        # DataFrame для хранения результатов
        result_queries_list = []

        for query in queries_name:
            # Определение языка введенного запроса
            detect_language = detect(query)

            # Флаг переводчика
            if not translator:
                # Транслитерация с использованием библиотеки transliterate
                latin_text = transliterate.translit(
                    query, detect_language, "en")
                result = process.extractOne(latin_text, city_names)
            else:
                # Использование переводчика
                translator = Translator(
                    provider="mymemory", from_lang=detect_language, to_lang="en")
                translation = translator.translate(query)
                result = process.extractOne(translation, city_names)

            # Добавление индекса
            answer_indx = self.df[self.df['asciiname'] == result[0]].index[0]

            result_queries_list.append({
                'query': query,
                'answer_indx': answer_indx,
                'score': result[1]
            })

        result_queries = pd.DataFrame(result_queries_list)

        return result_queries

    def find_similar(self, queries_name, top_k=1):
        '''
        Метод находит наиболее похожие города для заданных запросов.

        Параметры:
        - queries_name: Список запросов (названий городов) для поиска похожих.
        - top_k: Количество наиболее похожих результатов для каждого запроса (по умолчанию 1).

        Возвращаемое значение:
        - DataFrame с информацией о наиболее похожих результатах для каждого запроса.
        '''

        # Ограничение top_k, чтобы не превышать размер корпуса
        top_k = min(top_k, len(self.corpus_embeddings))

        # Создание пустого DataFrame для хранения результатов
        result_queries = pd.DataFrame(
            columns=['query', 'answer_indx', 'score'])

        # Итерация по каждому запросу
        for query in queries_name:
            # Получение эмбеддинга для текущего запроса
            query_embedding = self.model.encode(query, convert_to_tensor=True)

            # Использование cosine-similarity и torch.topk для поиска наивысших 5 оценок
            cos_scores = util.cos_sim(
                query_embedding, self.corpus_embeddings)[0]
            top_results = torch.topk(cos_scores, k=top_k)

            # Создание DataFrame с колонкой 'score'
            some_df = pd.DataFrame(columns=['query', 'answer_indx', 'score'])

            # Вставка массива в столбец 'score'
            some_df['answer_indx'] = np.array(top_results[1])
            some_df['score'] = np.array(top_results[0])

            # Заполнение столбца 'query' значением из запроса на всю длину других колонок
            some_df['query'] = [query] * len(top_results[0])

            # Конкатенация DataFrame с результатами
            result_queries = pd.concat([result_queries, some_df])

        return result_queries

    def get_geoname(self, queries_name, to_dict=False, how='tr',
                    top_k=1, translator=False, save_sql=False, sql_name='result_queries', if_exists='append'):
        '''
        Получение информации о наиболее похожих городах для заданных запросов.

        Параметры:
        - queries_name: Список запросов (названий городов).
        - to_dict: Флаг, указывающий, следует ли возвращать результат в виде словаря (по умолчанию False).
        - how: Способ поиска ('tr' для использования трансформера, 'lev' для расстояния Левенштейна).
        - top_k: Количество наиболее похожих результатов для каждого запроса (по умолчанию 1).
        - translator: Флаг использования переводчика (по умолчанию False).
        - save_sql: Флаг для сохранения результатов в базу данных (по умолчанию False).
        - sql_name: Название таблицы для сохранения результатов (по умолчанию 'result_queries').
        - if_exists: Стратегия при существующих записях в базе данных ('fail', 'replace' или 'append') (по умолчанию 'append').

        Возвращаемое значение:
        - DataFrame или словарь с информацией о наиболее похожих результатах для каждого запроса.
        '''

        if how == 'tr':
            # Проверка наличия модели для трансформера
            if not self.use_transformer:
                raise ValueError(
                    "Отсутствует модель, инициализируйте модель для использования")
            else:
                # Получение результатов с использованием трансформера
                result_queries = self.find_similar(queries_name, top_k)
                # Выбор соответствующих данных из DataFrame
                geoname = self.df.loc[result_queries['answer_indx'], [
                    'geonameid', 'name', 'region', 'country']].reset_index(drop=True)
                result_queries = result_queries.reset_index(drop=True)
                geoname['score'], geoname['query'] = result_queries['score'], result_queries['query']

                # Сохранение результатов в базу данных, если флаг установлен
                if save_sql:
                    geoname.to_sql(sql_name, con=self.engine,
                                   index=False, if_exists=if_exists)

                # Возвращение данных в формате DataFrame или словаря
                if to_dict:
                    return geoname.to_dict(orient='records')
                else:
                    return geoname
        elif how == 'lev' or not self.use_transformer:
            # Получение результатов с использованием расстояния Левенштейна
            result_queries = self.get_lev_distance(queries_name, translator)
            # Выбор соответствующих данных из DataFrame
            geoname = self.df.loc[result_queries['answer_indx'], [
                'geonameid', 'name', 'region', 'country']].reset_index(drop=True)
            result_queries = result_queries.reset_index(drop=True)
            geoname['score'], geoname['query'] = result_queries['score'], result_queries['query']

            # Сохранение результатов в базу данных, если флаг установлен
            if save_sql:
                geoname.to_sql(sql_name, con=self.engine,
                               index=False, if_exists=if_exists)

            # Возвращение данных в формате DataFrame или словаря
            if to_dict:
                return geoname.to_dict(orient='records')
            else:
                return geoname
        else:
            # Обработка неверного значения параметра 'how'
            raise ValueError(
                "Указанный способ 'how' не в списке ['lev', 'tr']")
