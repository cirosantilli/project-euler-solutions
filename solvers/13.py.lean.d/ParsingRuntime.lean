import solvers.«13.py.lean.d».FormattingRuntime
import lang.py.FormalSW.HeapRuntimeCostSumGenerators
import lang.py.FormalSW.IntegerParsing

namespace solvers.«13».runtimeCost
open FormalSW.PyAst HeapCalls HeapCalls.Runtime
private abbrev program := Solvers.P13Ast.module

def summation : Expr :=
  match executableBody ((findFunctionIn program.body "first_digits").getD ([], [])).2 with
  | .assign _ expression :: _ => expression
  | _ => .constant .none

def parsingElement : Expr :=
  match summation with | .sumGenerator element _ _ => element | _ => .constant .none

theorem parsing_element (heap : Heap) (env : HeapCalls.Env) (n : Int)
    (lookup : env "s" = some (.scalar (.string (toString n)))) :
    EvaluatesIn .word program env parsingElement heap (IntegerListWork.binarySize n+7)
      (.scalar (.int n)) heap := by
  have parsed : primitive (.builtin .int) [.scalar (.string (toString n))] heap =
      some (.ok (.scalar (.int n)), heap) := by
    change scalarResult (Calls.integerConversion [.string (toString n)]) heap = _
    rw [Calls.integerConversion_int]
    rfl
  have run := EvaluatesIn.builtin_single (globals := []) (arithmetic := .word) (module := program)
    (builtin := .int) (heap := heap) (work := (toString n).length+1)
    (by intro h; cases h) (by intro h; cases h) (EvaluatesIn.lookup lookup) parsed rfl
  exact run.mono (by have width := IntegerListWork.decimal_value_length n; omega)

theorem summation_cost (heap : Heap) (address : Nat) (items : List Int) (env : HeapCalls.Env)
    (found : heap[address]? = some (.list (texts items)))
    (lookup : env "nums" = some (.ref address)) :
    EvaluatesIn .word program env summation heap (19*IntegerListWork.size items+17)
      (.scalar (.int (items.foldl Int.add 0))) heap := by
  let localEnv := setLocal (comprehensionEnv env [("s", .name "nums", [])])
    sumAccumulator (.scalar (.int 0))
  have loop := sum_sequence_loop_costs (module := program) (env := localEnv) (initial := 0)
    (source := .ref address) (name := "s") (element := parsingElement)
    (values := items) (encode := fun n => .scalar (.string (toString n))) (number := id)
    items 0 (fun n => IntegerListWork.binarySize n+7) (by decide)
    (by simp [sequenceValue, Mutable.sequence, found, texts])
    (by intro name position body otherwise following env stack; simp [charge, found, PrimitiveWork.integer])
    (fun env n lookup => parsing_element heap env n lookup) rfl (by simp [localEnv, setLocal])
  have run := EvaluatesIn.sum_generator (module := program) (env := env)
    (iterable := .name "nums") (items := texts items) (size := items.length)
    (by intro expressions h; cases h) (EvaluatesIn.lookup lookup)
    (by simp [Mutable.sequence, found]) (by simp [sequenceLength, found, texts])
    EvaluatesIn.constant loop
  have work : ∀ values : List Int, (values.map (fun n => IntegerListWork.binarySize n+7+12)).sum ≤
      19*IntegerListWork.size values := by
    intro values
    induction values with
    | nil => simp [IntegerListWork.size]
    | cons head tail ih =>
        simp only [List.map_cons, List.sum_cons, IntegerListWork.size_cons]
        omega
  apply run.mono
  dsimp only
  have bound := work items
  omega

end solvers.«13».runtimeCost
