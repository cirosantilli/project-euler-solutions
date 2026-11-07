import ProjectEulerSolutions.Lib.MonopolyRanking
import ProjectEulerSolutions.Lib.HeapCollection
import lang.py.FormalSW.SumCalls

namespace ProjectEulerSolutions.MonopolyOutput
open FormalSW.PyAst HeapCalls
set_option Elab.async false

def encode {n : Nat} (values : Fin n → Int) : List Mutable.Value :=
  List.ofFn fun index => .scalar (.int (values index))

theorem square_window (values : Fin 120 → Int) (square : Fin 40) :
    ((encode values).drop (square.val * 3)).take 3 = encode (fun doubles : Fin 3 => values (Monopoly.stateIndex (square, doubles))) := by
  apply List.ext_getElem
  · simp only [encode, List.length_take, List.length_drop, List.length_ofFn]
    have inside := square.isLt
    omega
  · intro index left right
    simp only [encode, List.getElem_take, List.getElem_drop, List.getElem_ofFn]
    congr 3
    apply Fin.ext
    rw [Monopoly.stateIndex_val]
    simp [Monopoly.encode, Nat.add_comm]

def squareSlice : Expr := .subscript (.name "weights")
  (.slice (some (.binOp (.name "pos") .mult (.constant (.int 3))))
    (some (.binOp (.binOp (.name "pos") .mult (.constant (.int 3))) .add (.constant (.int 3)))) none)

def squareSum : Expr := .sumSequence squareSlice (.constant (.int 0))

theorem slice_result (module : Module) (values : Fin 120 → Int) (square : Fin 40)
    (heap : Mutable.Heap) (address : Nat) (found : heap[address]? = some (.list (encode values)))
    (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getWeights : env "weights" = some (.ref address)) (getSquare : env "pos" = some (.scalar (.int square.val))) :
    Evaluates module env squareSlice heap (.ref heap.length)
      (heap ++ [.list (encode (fun doubles : Fin 3 => values (Monopoly.stateIndex (square, doubles))))]) globals := by
  have start : Evaluates module env (.binOp (.name "pos") .mult (.constant (.int 3))) heap
      (.scalar (.int (square.val * 3 : Nat))) heap globals :=
    Evaluates.binary (.lookup getSquare) Evaluates.constant (by simp [Mutable.binary, scalarBinary])
  have stop : Evaluates module env (.binOp (.binOp (.name "pos") .mult (.constant (.int 3))) .add (.constant (.int 3))) heap
      (.scalar (.int (square.val * 3 + 3 : Nat))) heap globals :=
    Evaluates.binary start Evaluates.constant (by simp [Mutable.binary, scalarBinary])
  apply Evaluates.slice (.cons (.lookup getWeights) (.cons start (.cons stop (.cons Evaluates.constant .nil))))
  have operation := Mutable.getSlice_list_interval heap address (encode values) (square.val * 3) (square.val * 3 + 3)
    (by omega) (by simp only [encode, List.length_ofFn]; have inside := square.isLt; omega) found
  simpa only [Nat.add_sub_cancel_left, square_window] using operation

/-- Each source slice sums precisely the three consecutive doubles states of its square. -/
theorem sum_result (module : Module) (values : Fin 120 → Int) (square : Fin 40)
    (heap : Mutable.Heap) (address : Nat) (found : heap[address]? = some (.list (encode values)))
    (env : HeapCalls.Env) (globals : Mutable.Globals)
    (getWeights : env "weights" = some (.ref address)) (getSquare : env "pos" = some (.scalar (.int square.val))) :
    Evaluates module env squareSum heap (.scalar (.int (Monopoly.squareWeight values square)))
      (heap ++ [.list (encode (fun doubles : Fin 3 => values (Monopoly.stateIndex (square, doubles))))]) globals := by
  have sliced := slice_result module values square heap address found env globals getWeights getSquare
  have summed := Evaluates.sum_integers (module := module) (globals := globals) (initial := 0)
    (values := List.ofFn fun doubles : Fin 3 => values (Monopoly.stateIndex (square, doubles)))
    (.cons sliced (.cons Evaluates.constant .nil))
    (by simp [Mutable.sequence, encode])
  have total : (List.ofFn fun doubles : Fin 3 => values (Monopoly.stateIndex (square, doubles))).sum =
      (List.ofFn fun doubles : Fin 3 => values (Monopoly.stateIndex (square, doubles))).foldl Int.add 0 := List.sum_eq_foldl
  rw [← total, List.sum_ofFn] at summed
  exact summed

end ProjectEulerSolutions.MonopolyOutput
