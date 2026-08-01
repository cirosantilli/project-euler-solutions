import ProjectEulerStatements.P8
namespace ProjectEulerSolutions.P8

def hasZero : List Nat -> Bool
  | [] => false
  | x :: xs => x == 0 || hasZero xs

def productLoop (k : Nat) : List Nat -> List Nat
  | [] => []
  | digits@(_ :: xs) =>
      if digits.length < k then
        []
      else
        let products := productLoop k xs
        let window := digits.take k
        if hasZero window then
          products
        else
          ProjectEulerStatements.P8.listProduct window :: products

def maxAdjacentProduct (digits : List Nat) (k : Nat) : Nat :=
  if k == 0 then
    if digits.isEmpty then 0 else 1
  else
    ProjectEulerStatements.P8.listMax (productLoop k digits)


def solve (digits : List Nat) (k : Nat) : Nat :=
  maxAdjacentProduct digits k

example : solve ProjectEulerStatements.P8.digits1000 4 = 5832 := by
  native_decide
end ProjectEulerSolutions.P8
