import «.formalsw».solvers.«74.py.ast»
import ProjectEulerSolutions.Lib.HeapCollection
import Mathlib.Data.Nat.Factorial.Basic

/-! The optimized module creates its factorial table by executing the actual comprehension. -/
namespace solvers.«74».factorials
open FormalSW.PyAst HeapCalls
open ProjectEulerSolutions.HeapCollection
set_option Elab.async false

abbrev program := Solvers.P74Ast.module
abbrev number (n : Nat) : Mutable.Value := .scalar (.int n)
abbrev element : Expr := .call (.name "factorial") [.name "digit"]
abbrev tableExpr : Expr := .listComp element [("digit", .call (.name "range")
  [.constant (.int 10)], [])]

def values : List Mutable.Value := (List.range 10).map (fun n => number (Nat.factorial n))
def code : List Stmt := (executableBody program.body).drop 1 |>.take 1
def heap (caller : Mutable.Heap) : Mutable.Heap := caller ++ [.list values]
def env (caller : Mutable.Heap) : HeapCalls.Env :=
  setLocal (fun _ => none) "FACTORIALS" (.ref caller.length)
def globals (caller : Mutable.Heap) : Mutable.Globals := [("FACTORIALS", .ref caller.length)]

theorem code_source : code = [.assign [.name "FACTORIALS"] tableExpr] := rfl

private theorem factorial_eq (n : Nat) : Math.factorial n = Nat.factorial n := by
  induction n with
  | zero => rfl
  | succ n ih => simp [Math.factorial, Nat.factorial_succ, ih]

private theorem factorial_result (n : Nat) (caller : Mutable.Heap) (locals : HeapCalls.Env)
    (getDigit : locals "digit" = some (number n)) :
    Evaluates program locals element caller (number (Nat.factorial n)) caller := by
  apply Evaluates.library_call (.cons (Evaluates.lookup getDigit) .nil) rfl
  change some (Except.map Mutable.Value.scalar (Math.factorialCall [.int n]), caller) = _
  rw [Math.factorialCall_natural, factorial_eq]
  rfl

private theorem table_loop (first count stop : Nat) (bounds : first + count = stop)
    (caller : Mutable.Heap) (acc : List Mutable.Value) (locals : HeapCalls.Env) :
    ∃ finalEnv, BlockContinues program locals
      [.rangeStepLoop "digit" first stop 1 [.comprehensionAppend caller.length element] []]
      (working caller acc []) finalEnv
      (working caller (acc ++ (List.range' first count).map
        (fun n => number (Nat.factorial n))) []) := by
  induction count generalizing first acc locals with
  | zero =>
      refine ⟨locals, ?_⟩
      simp only [List.range'_zero, List.map_nil, List.append_nil]
      exact BlockContinues.range_done (by simp [Calls.rangeContinues]; omega) .nil
  | succ count ih =>
      let entered := setLocal locals "digit" (number first)
      have evaluated := factorial_result first (working caller acc []) entered
        (by simp [entered, setLocal])
      have appended := emit caller acc [] [] (number (Nat.factorial first))
        (by simpa only [List.append_nil] using evaluated)
      obtain ⟨finalEnv, completed⟩ := ih (first + 1) (by omega)
        (acc ++ [number (Nat.factorial first)]) entered
      refine ⟨finalEnv, ?_⟩
      apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
      simpa only [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt,
        Bool.false_or, Bool.false_eq_true, if_false, List.range'_succ, List.map_cons,
        List.append_assoc, List.singleton_append, Int.natCast_add, Int.natCast_one,
        List.nil_append] using appended.append completed

theorem table_result (caller : Mutable.Heap) (locals : HeapCalls.Env) :
    Evaluates program locals tableExpr caller (.ref caller.length) (heap caller) := by
  obtain ⟨finalEnv, completed⟩ := table_loop 0 10 10 rfl caller []
    (comprehensionEnv locals [("digit", .call (.name "range") [.constant (.int 10)], [])])
  apply Evaluates.list_comp_range (.cons Evaluates.constant .nil) rfl
  simpa only [working, values, heap, List.nil_append, List.range_eq_range'] using completed

/-- Initialization allocates the table after the caller's objects and exports its actual address. -/
theorem executes (caller : Mutable.Heap) :
    BlockContinues program (fun _ => none) code caller (env caller) (heap caller) := by
  rw [code_source]
  exact BlockContinues.assign (table_result caller (fun _ => none))

theorem exports (caller : Mutable.Heap) (name : String) :
    env caller name = (globals caller).lookup name := by
  by_cases same : name = "FACTORIALS"
  · subst name
    simp [env, globals, setLocal]
  · simp [env, globals, setLocal, List.lookup, same, beq_eq_false_iff_ne.mpr same]

theorem found (caller : Mutable.Heap) :
    (heap caller)[caller.length]? = some (.list values) := by simp [heap]

theorem value_at (digit : Nat) (small : digit < 10) :
    values[digit]? = some (number (Nat.factorial digit)) := by
  simp [values, List.getElem?_range small]

end solvers.«74».factorials
