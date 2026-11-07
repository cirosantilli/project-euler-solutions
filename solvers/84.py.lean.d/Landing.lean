import solvers.«84.py.lean.d».Transport
import lang.py.FormalSW.KeyDictionaryCalls
import lang.py.FormalSW.MembershipCalls

namespace solvers.«84».resolve_landing
open FormalSW.PyAst HeapCalls
set_option Elab.async false

private abbrev number (value : Int) : Mutable.Value := .scalar (.int value)
private abbrev args : List Arg := [⟨"square", none, none⟩]
private abbrev body : List Stmt := ((findFunctionIn program.body "resolve_landing").getD ([], [])).2

def singletonHeap (heap : Mutable.Heap) (square : Int) (jailed : Bool) : Mutable.Heap :=
  heap ++ [.tuple [number square, .scalar (.bool jailed)],
    .keyDict [(.tuple [.scalar (.int square), .scalar (.bool jailed)], .ref heap.length, number 256)]]

private theorem singleton (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals)
    (expression : Expr) (square : Int) (jailed : Bool)
    (evaluated : Evaluates program env expression heap (number square) heap globals) :
    Evaluates program env (.dict [(.tuple [expression, .constant (.bool jailed)], .constant (.int 256))])
      heap (.ref (heap.length + 1)) (singletonHeap heap square jailed) globals := by
  have key := Evaluates.tuple (.cons evaluated (.cons (Evaluates.constant (value := .bool jailed)) .nil)) (by simp)
  have hashed := Mutable.hashKey_tuple_scalars
    (heap ++ [.tuple [number square, .scalar (.bool jailed)]]) heap.length [.int square, .bool jailed] (by simp)
  simpa [singletonHeap, List.append_assoc] using Evaluates.dictionary_singleton key (Evaluates.constant (value := .int 256)) hashed

/-- Ordinary squares stay put; the two tested card-location tuples are retained as temporaries. -/
theorem ordinary_result (square : Int) (heap : Mutable.Heap) (globals : Mutable.Globals := [])
    (notJail : square ≠ 30) (notCommunity : square ∉ [2, 17, 33]) (notChance : square ∉ [7, 22, 36]) :
    ReturnsValue program "resolve_landing" [number square] heap
      (.ref (heap.length + 3))
      (singletonHeap (heap ++ [.tuple [number 2, number 17, number 33],
        .tuple [number 7, number 22, number 36]]) square false) globals := by
  have localSquare : functionEnv args body [number square] globals "square" = some (number square) := by
    cases globals <;> rfl
  apply BlockReturns.function (args := args) (body := body) rfl rfl
  apply BlockReturns.branch (flag := false) (value := .scalar (.bool false)) (middle := heap)
  · apply Evaluates.compare (Evaluates.lookup localSquare) Evaluates.constant
    simp [HeapCalls.compare, scalarCompare, evalCmpOp, notJail]
  · rfl
  · apply BlockReturns.branch (flag := false) (value := .scalar (.bool false))
      (middle := heap ++ [.tuple [number 2, number 17, number 33]])
    · apply Evaluates.membership (.cons (Evaluates.lookup localSquare) (.cons
        (Evaluates.tuple (.cons Evaluates.constant (.cons Evaluates.constant (.cons Evaluates.constant .nil))) (by simp)) .nil))
      simpa [notCommunity] using membership_tuple_ints
        (heap ++ [.tuple [number 2, number 17, number 33]]) heap.length square [2, 17, 33] (by simp) false
    · rfl
    · apply BlockReturns.branch (flag := false) (value := .scalar (.bool false))
        (middle := (heap ++ [.tuple [number 2, number 17, number 33]]) ++ [.tuple [number 7, number 22, number 36]])
      · apply Evaluates.membership (.cons (Evaluates.lookup localSquare) (.cons
          (Evaluates.tuple (.cons Evaluates.constant (.cons Evaluates.constant (.cons Evaluates.constant .nil))) (by simp)) .nil))
        simpa [notChance] using membership_tuple_ints
          (heap ++ [.tuple [number 2, number 17, number 33], .tuple [number 7, number 22, number 36]])
          (heap.length + 1) square [7, 22, 36] (by simp) false
      · rfl
      · apply BlockReturns.return_value
        convert singleton _
          ((heap ++ [.tuple [number 2, number 17, number 33]]) ++ [.tuple [number 7, number 22, number 36]])
          globals (.name "square") square false (Evaluates.lookup localSquare) using 1 <;>
          simp [List.append_assoc, Nat.add_assoc]

end solvers.«84».resolve_landing
