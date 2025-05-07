from app.domain.service.calculator import Calculator
from fastapi import FastAPI
import tensorflow as tf

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Calculator API"}

@app.get("/run-sample")
async def run_sample():
    calculator = Calculator()
    calculator.sample()
    return {"message": "Sample run completed"}

def test_sample():
    # TestCalc 인스턴스 생성
    calculator = Calculator()
    
    # sample 메소드 호출
    calculator.sample()
    
if __name__ == '__main__':

    def print_menu():
        print('0. 종료')
        print('+')
        print('-')
        print('*')
        print('/')
        return input('메뉴 입력\n')
    
    service = Calculator()

    while 1:
        num1 = input('숫자1 입력\n')
        a = tf.constant(int(num1))
        menu = print_menu()
        num2 = input('숫자2 입력\n')
        b = tf.constant(int(num2))
        if menu == '+':
            print('a + b = {}'.format(service.plus(a, b)))
        if menu == '-':
            print('a - b = {}'.format(service.minus(a, b)))
        if menu == '*':
            print('a * b = {}'.format(service.multiple(a, b)))
        if menu == '/':
            print('a / b = {}'.format(service.div(a, b)))
        elif menu == '0':
            break