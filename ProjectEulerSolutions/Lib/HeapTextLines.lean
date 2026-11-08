import ProjectEulerSolutions.Lib.HeapCollection
import lang.py.FormalSW.StringParsingCalls

/-! Execute the stripped, nonempty line comprehension used by both Sudoku parsers. -/
namespace ProjectEulerSolutions.HeapTextLines
open FormalSW.PyAst HeapCalls
set_option Elab.async false

abbrev textValue (text : String) : Mutable.Value := .scalar (.string text)
def selected (line : String) : List String :=
  if Strings.stripWhitespace line = "" then [] else [Strings.stripWhitespace line]
def clean (lines : List String) : List String := lines.flatMap selected
def normalized (text : String) : List String := clean (Strings.splitLines text)
abbrev stripped (name : String) : Expr := .call (.attribute (.name name) "strip") []
abbrev expression (name : String) (source : Expr) : Expr :=
  .listComp (stripped name) [(name, .call (.attribute source "splitlines") [], [stripped name])]
abbrev body (name : String) (address : Nat) : List Stmt :=
  comprehensionFilters [stripped name] [.comprehensionAppend address (stripped name)]
abbrev work (base : Mutable.Heap) (output : List String) : Mutable.Heap :=
  HeapCollection.working base (output.map textValue) []

theorem collect (program : Module) (name : String) (base : Mutable.Heap) (output : List String) (line : String)
    (env : HeapCalls.Env) (globals : Mutable.Globals) (getLine : env name = some (textValue line)) :
    BlockContinues program env (body name base.length) (work base output) env
      (work base (output ++ selected line)) globals := by
  have tested : Evaluates program env (stripped name) (work base output)
      (textValue (Strings.stripWhitespace line)) (work base output) globals :=
    Evaluates.strip (Evaluates.lookup getLine) (by intro other same; cases same; simp [getLine])
  by_cases empty : Strings.stripWhitespace line = ""
  · simp only [selected, if_pos empty, List.append_nil]
    exact BlockContinues.branch (flag := false) tested (by simp [Mutable.truth, scalarTruth, empty]) .nil
  · simp only [selected, if_neg empty]
    apply BlockContinues.branch (flag := true) tested (by simp [Mutable.truth, scalarTruth, empty])
    have emitted := BlockContinues.comprehension_append tested (HeapCollection.found base (output.map textValue) [])
    simpa only [work, HeapCollection.store, List.map_append, List.map_cons, List.map_nil] using emitted

theorem loop (program : Module) (name : String) (base : Mutable.Heap) (address : Nat)
    (before remaining output : List String) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (found : base[address]? = some (.list ((before ++ remaining).map textValue))) :
    ∃ afterEnv, BlockContinues program env
      [.sequenceStepLoop name (.ref address) before.length (body name base.length) []]
      (work base output) afterEnv (work base (output ++ clean remaining)) globals := by
  induction remaining generalizing before output env with
  | nil =>
      refine ⟨env, ?_⟩
      simp only [clean, List.flatMap_nil, List.append_nil]
      apply BlockContinues.sequence_done (items := before.map textValue)
      · have kept := HeapCollection.prefix_found base [.list (output.map textValue)] address _ found
        simp [Mutable.sequence, sequenceValue, work, HeapCollection.working, kept]
      · simp
      · exact .nil
  | cons line rest ih =>
      let ready := setLocal env name (textValue line)
      have emitted := collect program name base output line ready globals (by simp [ready, setLocal])
      obtain ⟨afterEnv, continued⟩ := ih (before ++ [line]) (output ++ selected line) ready
        (by simpa only [List.append_assoc, List.singleton_append] using found)
      refine ⟨afterEnv, ?_⟩
      apply BlockContinues.sequence_step (items := (before ++ line :: rest).map textValue) (value := textValue line)
      · have kept := HeapCollection.prefix_found base [.list (output.map textValue)] address _ found
        simp [Mutable.sequence, sequenceValue, work, HeapCollection.working, kept]
      · simp
      · simpa only [Calls.loopContinuation, Calls.loopControl, Calls.loopControlStmt, body, comprehensionFilters,
          Bool.false_or, Bool.false_eq_true, if_false, clean, List.flatMap_cons, List.append_assoc,
          List.length_append, List.length_singleton] using emitted.append continued

/-- The only allocations are the split lines and the filtered result; caller objects remain unchanged. -/
theorem result (program : Module) (name : String) (source : Expr) (text : String)
    (heap middle : Mutable.Heap) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (evaluated : Evaluates program env source heap (textValue text) middle globals)
    (localName : ∀ query, source = .name query → (env query).isSome = true) :
    Evaluates program env (expression name source) heap (.ref (middle.length + 1))
      (middle ++ [.list ((Strings.splitLines text).map textValue), .list ((normalized text).map textValue)]) globals := by
  let base := middle ++ [.list ((Strings.splitLines text).map textValue)]
  let inner := comprehensionEnv env [(name, .call (.attribute source "splitlines") [], [stripped name])]
  have found : base[middle.length]? = some (.list ((Strings.splitLines text).map textValue)) := by simp [base]
  obtain ⟨afterEnv, scanned⟩ := loop program name base middle.length [] (Strings.splitLines text) [] inner globals
    (by simpa using found)
  have split := Evaluates.split_lines evaluated localName
  have collected : Evaluates program env (expression name source) heap (.ref base.length)
      (work base (normalized text)) globals := by
    apply Evaluates.list_comp_sequence (finalEnv := afterEnv) (items := (Strings.splitLines text).map textValue)
      (by intros; simp) split
    · simp [Mutable.sequence]
    · exact scanned
  simpa only [base, work, HeapCollection.working, List.length_append, List.length_singleton, List.append_assoc] using collected

end ProjectEulerSolutions.HeapTextLines
