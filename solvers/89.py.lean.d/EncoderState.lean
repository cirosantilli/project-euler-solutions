import solvers.«89.py.lean.d».ParserState
import lang.py.FormalSW.UnpackReturnCalls
import ProjectEulerSolutions.Lib.HeapCollection
import ProjectEulerSolutions.Lib.RomanChunks

namespace solvers.«89».int_to_min_roman
open FormalSW.PyAst HeapCalls ProjectEulerSolutions RomanNumerals
set_option Elab.async false
set_option maxHeartbeats 800000

abbrev code : List Stmt := executableBody ((findFunctionIn program.body "int_to_min_roman").getD ([], [])).2
abbrev loopCode : List Stmt := match code[2]'(by decide) with
  | .forIn _ _ body _ => body
  | _ => []

structure Entry where
  denomination : Nat
  token : String
  address : Nat

def Entry.model (entry : Entry) : Nat × List Char := (entry.denomination, entry.token.toList)
def Entry.reference (entry : Entry) : Mutable.Value := .ref entry.address

structure Entry.Valid (heap : Mutable.Heap) (entry : Entry) : Prop where
  positive : 0 < entry.denomination
  found : heap[entry.address]? = some (.tuple [number entry.denomination, textValue entry.token])

abbrev working (base : Mutable.Heap) (output : List String) (garbage : Mutable.Heap) : Mutable.Heap :=
  HeapCollection.working base (output.map textValue) garbage

abbrev dividedEnv (env : HeapCalls.Env) (n denomination : Nat) : HeapCalls.Env :=
  assignNames env ["k", "x"] [number (n / denomination), number (n % denomination)]

abbrev quotientObject (n denomination : Nat) : Mutable.Object :=
  .tuple [number (n / denomination), number (n % denomination)]

/-- Python's divmod allocates a tuple and assigns its quotient and remainder left to right. -/
theorem divide (n denomination : Nat) (positive : 0 < denomination)
    (heap : Mutable.Heap) (globals : Mutable.Globals) (env : HeapCalls.Env)
    (getN : env "x" = some (number n)) (getD : env "val" = some (number denomination)) :
    BlockContinues program env [loopCode[1]'(by decide)] heap (dividedEnv env n denomination)
      (heap ++ [quotientObject n denomination]) globals := by
  apply BlockContinues.unpack (items := [number (n / denomination), number (n % denomination)])
    (value := .ref heap.length) (finalHeap := heap ++ [quotientObject n denomination]) rfl
  · apply Evaluates.builtin (.cons (Evaluates.lookup getN) (.cons (Evaluates.lookup getD) .nil))
    have divided := Mutable.divmod_ints (n : Int) denomination (by omega) heap
    have quotient : (n : Int).fdiv denomination = ((n / denomination : Nat) : Int) := by
      rw [Int.fdiv_eq_ediv_of_nonneg _ (by omega)]
      rfl
    have remainder : (n : Int) - denomination * (n : Int).fdiv denomination = ((n % denomination : Nat) : Int) := by
      rw [Int.fdiv_eq_ediv_of_nonneg _ (by omega), ← Int.emod_def]
      rfl
    simpa only [quotient, remainder] using divided
  · simp [Mutable.sequence, quotientObject, Int.emod_def]
  · rfl

/-- A positive quotient appends one repeated string chunk to the existing result list. -/
theorem emit (token : String) (count : Nat) (positive : 0 < count)
    (base : Mutable.Heap) (output : List String) (garbage : Mutable.Heap) (globals : Mutable.Globals)
    (env : HeapCalls.Env) (getCount : env "k" = some (number count))
    (getToken : env "sym" = some (textValue token)) (getOutput : env "res" = some (.ref base.length)) :
    BlockContinues program env [loopCode[2]'(by decide)] (working base output garbage) env
      (working base (output ++ [String.join (List.replicate count token)]) garbage) globals := by
  apply BlockContinues.branch (flag := true) (Evaluates.lookup getCount)
  · simp [Mutable.truth, scalarTruth, show count ≠ 0 by omega]
  · apply BlockContinues.discard (value := .scalar .none)
    have appended := Evaluates.append (module := program) (globals := globals)
      (Evaluates.lookup getOutput) (HeapCollection.found base (output.map textValue) garbage)
      (Evaluates.binary (Evaluates.lookup getToken) (Evaluates.lookup getCount)
        (show Mutable.binary .mult (textValue token) (number count) (working base output garbage) =
          some (.ok (textValue (String.join (List.replicate count token))), working base output garbage) by
          rfl))
      (HeapCollection.found base (output.map textValue) garbage)
    simpa only [working, HeapCollection.store, List.map_append, List.map_cons, List.map_nil] using appended

/-- The final join observes every accumulated chunk without changing the heap. -/
theorem finish (base : Mutable.Heap) (output : List String) (garbage : Mutable.Heap)
    (globals : Mutable.Globals) (env : HeapCalls.Env) (getOutput : env "res" = some (.ref base.length)) :
    BlockReturns program env [code[3]'(by decide)] (working base output garbage)
      (textValue (String.ofList (output.flatMap String.toList))) (working base output garbage) globals := by
  apply BlockReturns.return_value
  apply Evaluates.join (values := [.ref base.length]) (afterArguments := working base output garbage) Evaluates.constant
  · exact .cons (Evaluates.lookup getOutput) .nil
  · simpa only [join_chunks] using primitive_join_list (working base output garbage) base.length "" output
      (HeapCollection.found base (output.map textValue) garbage)
  · intros name impossible
    cases impossible

end solvers.«89».int_to_min_roman
