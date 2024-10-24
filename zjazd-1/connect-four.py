"""
Gra w Connect Four

Zasady gry: https://pl.wikipedia.org/wiki/Czwórki

Autorzy:
Szymon Rosztajn, Mateusz Lech

Instrukcja przygotowania środowiska:
1. Zainstaluj wymagane biblioteki:
   pip install numpy easyAI
2. Uruchom grę:
   python connect_four.py

Wymagania:
- Python 3.x
- Biblioteka numpy
- Biblioteka easyAI
"""

from easyAI import TwoPlayerGame, AI_Player, Human_Player, Negamax
import numpy as np

BLUE = '\033[94m'
RED = '\033[91m'
ENDC = '\033[0m'

class ConnectFour(TwoPlayerGame):
    """
    Klasa reprezentująca grę Connect Four, rozszerzająca TwoPlayerGame z easyAI.
    """

    def __init__(self, players):
        """
        Inicjalizuje grę, ustawia planszę 6x7 i gracza zaczynającego.

        Parametry:
        players (list): Lista graczy (człowiek i AI).
        """
        self.players = players
        self.board = np.zeros((6, 7), dtype=int)
        self.current_player = 1

    def possible_moves(self):
        """
        Zwraca listę kolumn, do których można dodać znacznik (kolumny, które nie są pełne).

        Zwraca:
        list: Lista dostępnych kolumn (od 1 do 7).
        """
        return [str(i+1) for i in range(7) if self.board[0][i] == 0]

    def make_move(self, column):
        """
        Umieszcza znacznik gracza w podanej kolumnie.

        Parametry:
        column (str): Numer kolumny wybrany przez gracza.
        """
        column = int(column) - 1
        row = max([r for r in range(6) if self.board[r][column] == 0])
        self.board[row][column] = self.current_player

    def unmake_move(self, column):
        """
        Usuwa znacznik z podanej kolumny.

        Parametry:
        column (str): Numer kolumny z której cofamy ruch.
        """
        column = int(column) - 1
        row = min([r for r in range(6) if self.board[r][column] != 0])
        self.board[row][column] = 0

    def is_over(self):
        """
        Sprawdza, czy gra się zakończyła.

        Zwraca:
        bool: True, jeśli gra się zakończyła, w przeciwnym razie false.
        """
        return self.possible_moves() == [] or self.winner() is not None

    def show(self):
        """
        Wyświetla aktualny stan planszy z numerami kolumn i kolorowymi znacznikami.
        """
        print("\n Kolumny  1 2 3 4 5 6 7")
        for r in range(6):
            row_str = []
            for c in range(7):
                if self.board[r][c] == 1:
                    row_str.append(f"{BLUE}O{ENDC}")
                elif self.board[r][c] == 2:
                    row_str.append(f"{RED}O{ENDC}")
                else:
                    row_str.append(".")
            print("          " + " ".join(row_str))
        print("\n")

    def spot_winner(self, line):
        """
        Sprawdza, czy w linii są cztery te same znaczniki.

        Parametry:
        line (list): Lista 4 elementów reprezentujących znaczniki w danej linii.

        Zwraca:
        bool: True, jeśli w linii są cztery takie same znaczniki, w przeciwnym razie false.
        """
        return line == [self.current_player] * 4

    def winner(self):
        """
        Sprawdza wszystkie wiersze, kolumny i przekątne, czy któryś z graczy wygrał.

        Zwraca:
        int: Numer gracza, który wygrał, lub none, jeśli nikt nie wygrał.
        """
        for r in range(6):
            for c in range(4):
                if self.spot_winner([self.board[r][c + i] for i in range(4)]):
                    return self.current_player
        for r in range(3):
            for c in range(7):
                if self.spot_winner([self.board[r + i][c] for i in range(4)]):
                    return self.current_player
        for r in range(3):
            for c in range(4):
                if self.spot_winner([self.board[r + i][c + i] for i in range(4)]) or \
                   self.spot_winner([self.board[r + 3 - i][c + i] for i in range(4)]):
                    return self.current_player
        return None

    def scoring(self):
        """
        Funkcja punktowa - jeśli AI wygra, dostaje 100 punktow.

        Zwraca:
        int: 100, jeśli AI wygra, w przeciwnym razie 0.
        """
        return 100 if self.winner() == self.current_player else 0

if __name__ == "__main__":
    """
    Główna funkcja programu. Ustawia algorytm AI i rozpoczyna rozgrywkę między człowiekiem a AI.
    """
    ai_algo = Negamax(5)
    game = ConnectFour([Human_Player(), AI_Player(ai_algo)])
    game.play()

    if game.winner() == 1:
        print(f"{BLUE}You win!{ENDC}")
    elif game.winner() == 2:
        print(f"{RED}AI win!{ENDC}")
    else:
        print("Draw!")
