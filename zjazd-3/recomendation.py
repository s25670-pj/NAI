import argparse
import json
import operator

"""
Autorzy:Mateusz Lech, Szymon Rosztajn
Program pozwala na podstawie przygotowanej liscie filmow i osob zaproponowac jakie filmy wybrana osoba powinna obejrzec i ktore powinna omijac

Wymagania:
 - Przygotowana lista uzytkownikow, filmow i ich ocen w pliku JSON
 - Python3

Uruchomienie:
 python recomendation.py --user "Imie Nazwisko"
"""


def build_arg_parser():
    """
    Funkcja która obsługuje argumenty podane podczas włączania programu
    :return: obiekt zwierający wszystkie wspierane i podane argumenty
    """
    parser = argparse.ArgumentParser(description="Provide list of movies worth watching and movies that should be avoided")
    parser.add_argument('--user', dest='user', required=True,
                        help='User which should receive recommendation based on json database')
    return parser


def movie_recommendation(dataset, user):
    """
    Funkcja służąca do sprawdzania jakie filmy użytkownik powinien obejrzeć a jakich nie na podstawie danych zawartych w pliku JSON
    :param dataset: parametr danych z pliku JSON
    :param user: użytkownik dla którego będzie robiona weryfikacja
    :return: 2 listy filmowow - polecane i niepolecane
    """

    user_data = next((user_data for user_data in dataset if user_data['name'] == user), None)
    if not user_data:
        raise TypeError(f'Cannot find {user} in the dataset')

    watched_movies = {movie['title']: movie['rating'] for movie in user_data['movies']}
    all_movies = set()
    not_watched_movies = {}
    
    for user_data in dataset:
        for movie in user_data['movies']:
            all_movies.add(movie['title'])
            if movie['title'] not in watched_movies:
                if movie['title'] not in not_watched_movies:
                    not_watched_movies[movie['title']] = 0
                not_watched_movies[movie['title']] += movie['rating']

    recommended_movies = dict(sorted(not_watched_movies.items(), key=operator.itemgetter(1), reverse=True)[:5])
    avoided_movies = dict(sorted(not_watched_movies.items(), key=operator.itemgetter(1))[:5])


    print(f"5 rekomendacji dla {user}:")
    for movie in recommended_movies:
        print(f" - {movie}")

    print(f"\n5 antyrekomendacji dla {user}:")
    for movie in avoided_movies:
        print(f" - {movie}")


if __name__ == '__main__':
    """
    Glowny fragment programu wywolujacy wszystkie zaimplementowane funckje
    """
    args = build_arg_parser().parse_args()
    user = args.user
    db = 'data.json'

    with open(db, 'r', encoding="utf-8") as f:
        data = json.load(f)

    movie_recommendation(data, user)
