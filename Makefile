CC = gcc
CFLAGS = -ansi -Wall -Wextra -Werror -pedantic-errors

symnmf: symnmf.o symnmf.h
	$(CC) -o symnmf symnmf.o -lm

symnmf.o: symnmf.c
	$(CC) $(CFLAGS) -c symnmf.c

clean:
	rm -f *.o symnmf