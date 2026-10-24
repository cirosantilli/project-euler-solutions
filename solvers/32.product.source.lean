import solvers.«32.predicate.source»
import ProjectEulerSolutions.Lib.PandigitalProductSearch

set_option Elab.async false

namespace solvers.«32»
open FormalSW.PyAst FormalSW.PyAst.HeapCalls ProjectEulerSolutions.PandigitalProductSearch
private abbrev program := Solvers.P32Ast.module
private abbrev number (n : Nat) : Mutable.Value := .scalar (.int n)
private abbrev printed (n : Nat) : Mutable.Value := .scalar (.string (toString (n : Int)))

private theorem characters (n : Nat) (positive : 0 < n) :
    (toString (n : Int)).toList = (digits n).map Nat.digitChar := by
  simpa [toString, Int.repr, Nat.repr, digits] using
    ProjectEulerStatements.DecimalDigits.toDigits_eq_digitChars n positive

theorem product_predicate (width a b : Nat) (aPositive : 0 < a) (bPositive : 0 < b)
    (heap : Mutable.Heap) :
    ReturnsValue program "is_pandigital"
      [.scalar (.string (toString (a : Int) ++ toString (b : Int) ++ toString ((a * b : Nat) : Int))),
        number width] heap (.scalar (.bool (accepted width a b))) heap := by
  have rendered : String.ofList ((digits a ++ digits b ++ digits (a * b)).map Nat.digitChar) =
      toString (a : Int) ++ toString (b : Int) ++ toString ((a * b : Nat) : Int) := by
    apply String.toList_injective
    simp only [String.toList_ofList, List.map_append, String.toList_append,
      characters a aPositive, characters b bPositive,
      characters (a * b) (Nat.mul_pos aPositive bPositive)]
  have decimal : ∀ digit ∈ digits a ++ digits b ++ digits (a * b), digit < 10 := by
    intro digit member
    simp only [List.mem_append] at member
    rcases member with (member | member) | member
    · exact decimal_digits a digit member
    · exact decimal_digits b digit member
    · exact decimal_digits (a * b) digit member
  simpa only [accepted, rendered] using
    is_pandigital.result (globals := []) width (digits a ++ digits b ++ digits (a * b)) decimal heap

def productTest : Expr := .call (.name "is_pandigital")
  [.binOp (.binOp (.builtinCall .str [.name "a"]) .add (.builtinCall .str [.name "b"]))
    .add (.builtinCall .str [.name "product"]), .name "width"]

theorem product_test (width a b : Nat) (aPositive : 0 < a) (bPositive : 0 < b)
    (heap : Mutable.Heap) (env : HeapCalls.Env)
    (getA : env "a" = some (number a)) (getB : env "b" = some (number b))
    (getProduct : env "product" = some (number (a * b)))
    (getWidth : env "width" = some (number width)) :
    Evaluates program env productTest heap (.scalar (.bool (accepted width a b))) heap := by
  apply Evaluates.call
    (.cons (Evaluates.binary (av := .scalar (.string (toString (a : Int) ++ toString (b : Int))))
      (bv := printed (a * b)) (middle := heap) (afterRight := heap) (finalHeap := heap)
      (Evaluates.binary (av := printed a) (bv := printed b)
        (middle := heap) (afterRight := heap) (finalHeap := heap)
        (Evaluates.builtin (.cons (Evaluates.lookup getA) .nil) rfl)
        (Evaluates.builtin (.cons (Evaluates.lookup getB) .nil) rfl) rfl)
      (Evaluates.builtin (.cons (Evaluates.lookup getProduct) .nil) rfl) rfl)
      (.cons (Evaluates.lookup getWidth) .nil))
  exact product_predicate width a b aPositive bPositive heap

end solvers.«32»
