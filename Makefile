CC=gcc
CFLAGS=-g -Wall
# CFLAGS=-O3
SRC=src
OBJ=obj
BIN=bin
TARGET=program
SRCS=$(wildcard $(SRC)/*.c)
OBJS=$(patsubst $(SRC)/%.c, $(OBJ)/%.o, $(SRCS))

LIBS= -lcurl -lsqlite3 -lgsl -lm -lgslcblas

$(info SRCS: $(SRCS))
$(info OBJS:$(OBJS))
$(info ---)

all: $(BIN)/$(TARGET) 

$(OBJ)/%.o: $(SRC)/%.c 
	$(CC) $(CFLAGS) -c $< -o $@

$(BIN)/$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) $(LIBS) -o $@

clean: 
	rm -f $(BIN)/* $(OBJ)/* *.txt *.plt *.db
