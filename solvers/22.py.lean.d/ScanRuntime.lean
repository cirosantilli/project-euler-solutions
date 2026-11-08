import solvers.«22.py.lean.d».NameRuntime
import lang.py.FormalSW.HeapRuntimeCostEnumeration
import lang.py.FormalSW.StringListWork

namespace solvers.«22».runtimeCost.Scan
open FormalSW.PyAst HeapCalls HeapCalls.Runtime

abbrev program := Solvers.P22Ast.module
abbrev number := Name.number

def definition : List Arg × List Stmt :=
  (findFunctionIn program.body "solve").getD ([], [])

def body : List Stmt :=
  match ((executableBody definition.2)[2]? : Option Stmt) with
  | some (.forIn _ _ body _) => body
  | _ => []

theorem body_eq : body =
    [.augAssign (.name "total") .add (.binOp (.name "i") .mult
      (.call (.name "name_value") [.name "nm"]))] := rfl

theorem body_continuation (next : Stmt) : Calls.loopContinuation body next = next := rfl

/-- One actual source accumulation, with a signed counter and total. -/
theorem item_cost (counter total : Int) (text : String) (heap : Mutable.Heap) (env : HeapCalls.Env)
    (index : env "i" = some (number counter))
    (name : env "nm" = some (.scalar (.string text)))
    (accumulated : env "total" = some (number total)) :
    BlockContinuesIn .word program env body heap (31 * text.length + 35)
      (setLocal env "total" (number (total + counter * Name.value text))) heap := by
  have called := EvaluatesIn.call_single (EvaluatesIn.lookup name) (Name.function_run text heap)
  have multiplied : EvaluatesIn .word program env
      (.binOp (.name "i") .mult (.call (.name "name_value") [.name "nm"])) heap
      (31 * text.length + 29) (number (counter * Name.value text)) heap := by
    have run := EvaluatesIn.binary (op := .mult) (work := 1)
      (value := number (counter * Name.value text)) (EvaluatesIn.lookup index) called
      (by simp [Mutable.binary, scalarBinary, number, Name.number]) rfl
    exact run.mono (by omega)
  rw [body_eq]
  exact BlockContinuesIn.augAssign (work := 1) accumulated multiplied
    (by simp [inplace, Mutable.binary, scalarBinary, number, Name.number]) rfl

/-- Retain individual string lengths through the complete enumeration. -/
theorem loop_cost (values remaining : List String) (position address : Nat) (counter total : Int)
    (heap : Mutable.Heap) (env : HeapCalls.Env)
    (found : heap[address]? = some (.list (StringList.encode values)))
    (suffix : values.drop position = remaining)
    (accumulated : env "total" = some (number total)) :
    ∃ (finalTotal : Int) (finalEnv : HeapCalls.Env),
      finalEnv "total" = some (number finalTotal) ∧
      BlockContinuesIn .word program env
        [.enumerateStepLoop "i" "nm" (.ref address) position counter body []] heap
        ((remaining.map (fun text => 31 * text.length + 42)).sum + 7) finalEnv heap := by
  induction remaining generalizing position counter total env with
  | nil =>
      refine ⟨total, env, accumulated, ?_⟩
      have missing : values[position]? = none := by
        simpa [suffix] using (List.head?_drop (l := values) (i := position)).symm
      apply BlockContinuesIn.list_enumerate_done found
        (by simp [StringList.encode, missing]) BlockContinuesIn.nil
  | cons text rest ih =>
      have selected : values[position]? = some text := by
        simpa [suffix] using (List.head?_drop (l := values) (i := position)).symm
      let indexed := setLocal (setLocal env "i" (number counter)) "nm" (.scalar (.string text))
      have index : indexed "i" = some (number counter) := by simp [indexed, setLocal]
      have name : indexed "nm" = some (.scalar (.string text)) := by simp [indexed, setLocal]
      have totalSaved : indexed "total" = some (number total) := by
        simpa [indexed, setLocal] using accumulated
      have item := item_cost counter total text heap indexed index name totalSaved
      obtain ⟨finalTotal, finalEnv, finalSaved, tail⟩ := ih (position + 1) (counter + 1)
        (total + counter * Name.value text)
        (setLocal indexed "total" (number (total + counter * Name.value text)))
        (by simp [List.drop_add_one_eq_tail_drop, suffix]) (by simp [setLocal])
      refine ⟨finalTotal, finalEnv, finalSaved, ?_⟩
      have run := BlockContinuesIn.list_enumerate_step (index := "i") (name := "nm")
        (value := .scalar (.string text))
        (position := position) (counter := counter) (body := body) (otherwise := []) (env := env)
        found (by simp [StringList.encode, selected])
        (by simpa only [body_continuation, indexed, number, Name.number] using item.append tail)
      exact run.mono (by simp only [List.map_cons, List.sum_cons]; omega)

end solvers.«22».runtimeCost.Scan
