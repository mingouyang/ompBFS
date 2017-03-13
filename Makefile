CC=icc
CFLAGS=-Wall -O3 -xHost -ipo -qopenmp

%: %.c
	$(CC) $(CFLAGS) -o $@ $<
