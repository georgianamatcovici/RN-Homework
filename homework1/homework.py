import pathlib
import copy

def load_system(path: pathlib.Path) -> tuple[list[list[float]], list[float]]:
    result=[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [0, 0, 0]
    file=open(path, "r")
    line_number=0
    for line in file:
        line=line.strip()
        variable=0
        coefficient=0
        sign=1
        i=0
        while(i<len(line)):
            if line[i].isdigit():
                j=i
                while(i<len(line) and (line[i].isdigit() or line[i]=='.')):
                    i+=1
                coefficient=float(line[j:i])
                if variable<=2:
                    result[0][line_number][variable]=sign*coefficient
                    print(sign*coefficient)
                    variable+=1
                    sign=1
                    coeffiecient=0
                else:
                     result[1][line_number]=sign*coefficient
                     print(sign*coefficient)
                
            elif line[i]=='+':
                    sign=1
            elif line[i]=='-':
                    sign=-1
            elif line[i]=='x' or line[i]=='y' or line[i]=='z':
                 result[0][line_number][variable]=sign
                 print(sign)
                 variable+=1
                 sign=1
                 coeffiecient=0
            i+=1
        line_number+=1

    return result

def determinant(matrix: list[list[float]]) -> float:
    first=(matrix[1][1]*matrix[2][2]-matrix[1][2]*matrix[2][1])
    second=(matrix[1][0]*matrix[2][2]-matrix[1][2]*matrix[2][0])
    third=(matrix[1][0]*matrix[2][1]-matrix[1][1]*matrix[2][0])
    return matrix[0][0]*first-matrix[0][1]*second+matrix[0][2]*third

def determinant2(matrix: list[list[float]]) -> float:
   return(matrix[0][0]*matrix[1][1]-matrix[1][0]*matrix[0][1])

def trace(matrix: list[list[float]]) -> float:
    result=0
    for i in range(0, len(matrix)):
         result+=matrix[i][i]
    return result
         
def norm(vector: list[float]) -> float:
    result=0
    for i in range(0, len(vector)):
         result+=(vector[i]**2)
    return result**0.5

def transpose(matrix: list[list[float]]) -> list[list[float]]:
    transpose_matrix= [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for i in range(0, len(matrix)):
         for j in range(0, len(matrix[i])):
              transpose_matrix[j][i]=matrix[i][j]
    return transpose_matrix

def multiply(matrix: list[list[float]], vector: list[float]) -> list[float]:
    n=len(matrix)
    print(n)
    p=len(vector)
    result=[0, 0, 0]
    for i in range (0, n):
         m=len(matrix[i])
         if m!=p:
              return None
         res=0
         for j in range(0, m):
              res=res+matrix[i][j]*vector[j]
         result[i]=res
    return result


def solve_cramer(matrix: list[list[float]], vector: list[float]) -> list[float]:
    n=len(matrix)
    det_matrix=determinant(matrix)
    det_x=0
    result=[0,0,0]
    if(len(matrix[0])!=n):
         return None
    if(len(vector)!=n):
         return None
    for i in range (0, n):
        matrix_copy=copy.deepcopy(matrix)
        for j in range(0, n):
              matrix_copy[j][i]=vector[j]
        print(matrix_copy)
        det_x=determinant(matrix_copy)
        print(det_x)
        result[i]=det_x/det_matrix
    return result

def minor(matrix: list[list[float]], i: int, j: int) -> list[list[float]]:
    result=[[0, 0], [0, 0]]
    n=len(matrix)
    for k in range(0, n):
         if(k!=i):
            for l in range(0, n):
                if(l!=j):
                    if(k<i):
                        x=k
                    else: x=k-1
                    if(l<j):
                        y=l
                    else: y=l-1
                    result[x][y]=matrix[k][l]
    return result
            
def cofactor(matrix: list[list[float]]) -> list[list[float]]:
    result= [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    n=len(matrix)
    for i in range(0, n):
         for j in range(0, n):
              result[i][j]=(-1)**(i+j)*determinant2(minor(matrix, i, j))
    return result


def adjoint(matrix: list[list[float]]) -> list[list[float]]:
    return transpose(cofactor(matrix))

def constant_multiply(matrix: list[list[float]], c:float) -> list[list[float]]:
    result=copy.deepcopy(matrix)
    n=len(matrix)
    for i in range(0, n):
         for j in range(0, n):
              print((matrix[i][j]))
              result[i][j]=c*matrix[i][j]
    return result
         
def solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    inverse_matrix=constant_multiply(adjoint(matrix),1/determinant(matrix))
    return multiply(inverse_matrix, vector)

        
A, B = load_system(pathlib.Path("system.txt"))
print(f"{A=} {B=}") 
print(f"{determinant(A)=}")
print(f"{trace(A)=}")
print(f"{norm(B)=}")
print(f"{transpose(A)=}")
print(f"{multiply(A, B)=}")
print(f"{solve_cramer(A, B)=}")
print(f"{solve(A, B)=}")