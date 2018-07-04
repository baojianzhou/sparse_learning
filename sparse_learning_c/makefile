OPENBLAS = /usr/lib/x86_64-linux-gnu/openblas/
INCLUDE = /usr/include/x86_64-linux-gnu/

TARGET_test_loss_func = ./bin/test_loss_func
TARGET_test_logistic_regression = ./bin/test_logistic_regression

CFLAGS = -lm -I${INCLUDE} -L${OPENBLAS} -O2 -lopenblas -lpthread -lgfortran

#$(TARGET_test_loss_func): $(src/loss_func.c src/loss_func.h src/test_loss_func.c)
#$(TARGET_test_logistic_regression): $(src/test_logistic_regression.c src/logistic_regression.h src/logistic_regression.c)
all:
	/bin/rm -f $(TARGET_test_loss_func)
	gcc $(src/loss_func.c src/loss_func.h src/test_loss_func.c) $(CFLAGS) -o $(TARGET_test_loss_func)
	gcc $(src/test_logistic_regression.c src/logistic_regression.h src/logistic_regression.c) $(CFLAGS) -o $(TARGET_test_logistic_regression)
clean:
	/bin/rm -f $(TARGET_test_loss_func)
	/bin/rm -f $(TARGET_test_logistic_regression)