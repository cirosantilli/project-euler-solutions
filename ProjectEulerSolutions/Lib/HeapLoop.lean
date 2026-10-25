import lang.py.FormalSW.HeapCalls
import ProjectEulerSolutions.Lib.HeapCollection
import ProjectEulerSolutions.Lib.HeapRangeFold

/-! Composing loop bodies with their actual `continue` boundary. -/
namespace ProjectEulerSolutions.HeapLoop
open FormalSW.PyAst FormalSW.PyAst.HeapCalls
variable {module : Module} {globals : Mutable.Globals}
variable {env middleEnv finalEnv : HeapCalls.Env} {body tail yes no : List Stmt}
variable {heap middleHeap finalHeap middle : Mutable.Heap} {test : Expr} {value : Mutable.Value}
universe u v

def Continues (module : Module) (env : HeapCalls.Env) (body : List Stmt) (heap : Mutable.Heap)
    (finalEnv : HeapCalls.Env) (finalHeap : Mutable.Heap) (globals : Mutable.Globals := []) : Prop :=
  ∀ next following anchors stack, Reaches module anchors
    ⟨.block (body ++ [.loopResume next] ++ following) env, stack, heap⟩
    ⟨.block (next :: following) finalEnv, stack, finalHeap⟩ globals

theorem nil (module : Module) (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals := []) :
    Continues module env [] heap env heap globals := fun _ _ _ _ => .next rfl .refl

theorem prepend (first : BlockContinues module env body heap middleEnv middleHeap globals)
    (rest : Continues module middleEnv tail middleHeap finalEnv finalHeap globals) :
    Continues module env (body ++ tail) heap finalEnv finalHeap globals := by
  intro next following anchors stack
  have joined :=
    (first (tail ++ [.loopResume next] ++ following) anchors stack).trans (rest next following anchors stack)
  simpa only [List.append_assoc] using joined

theorem branch (flag : Bool) (evaluated : Evaluates module env test heap value middle globals)
    (truth : Mutable.truth value middle = some (.ok flag, middle))
    (rest : Continues module env ((if flag then yes else no) ++ body) middle finalEnv finalHeap globals) :
    Continues module env (.ifThen test yes no :: body) heap finalEnv finalHeap globals := by
  intro next following anchors stack
  have resumed : Reaches module anchors
      ⟨.block ((if flag then yes else no) ++ (body ++ [.loopResume next] ++ following)) env, stack, middle⟩
      ⟨.block (next :: following) finalEnv, stack, finalHeap⟩ globals := by
    have executed := rest next following anchors stack
    simpa only [List.append_assoc] using executed
  exact .next rfl ((evaluated anchors _).trans (.next (by simp [step, run, truth]) resumed))

theorem skip (module : Module) (env : HeapCalls.Env) (body : List Stmt) (heap : Mutable.Heap)
    (target : ∀ next following, Calls.loopTarget (body ++ [.loopResume next] ++ following) = some (next, following))
    (globals : Mutable.Globals := []) :
    Continues module env (.continueLoop :: body) heap env heap globals := by
  intro next following anchors stack
  have selected : Calls.loopTarget (body ++ .loopResume next :: following) = some (next, following) := by
    simpa using target next following
  exact .next (by simp [step, selected]) .refl

/-- A finite sequence fold keeps the original iterator live across bodies that use `continue`. -/
theorem fold {α : Type u} {State : Type v} (module : Module) (name : String) (body : List Stmt)
    (control : Calls.loopControl body = true) (changed : List String) (written : name ∈ changed)
    (encode : α → Mutable.Value) (update : State → α → State)
    (Invariant : Mutable.Heap → HeapCalls.Env → State → Prop) (address : Nat) (values : List α)
    (bindInvariant : ∀ heap env state item, Invariant heap env state → item ∈ values →
      Invariant heap (setLocal env name (encode item)) state)
    (cell : ∀ heap env state item, Invariant heap env state → item ∈ values → env name = some (encode item) →
      ∃ extra finalEnv, Continues module env body heap finalEnv (heap ++ extra) globals ∧
        Invariant (heap ++ extra) finalEnv (update state item) ∧ HeapRangeFold.Frame changed env finalEnv)
    (remaining : List α) (position : Nat) (suffix : values.drop position = remaining)
    (heap : Mutable.Heap) (env : HeapCalls.Env) (state : State) (safe : Invariant heap env state)
    (found : heap[address]? = some (.list (values.map encode))) :
    ∃ extra finalEnv, BlockContinues module env [.sequenceStepLoop name (.ref address) position body []]
      heap finalEnv (heap ++ extra) globals ∧
      Invariant (heap ++ extra) finalEnv (remaining.foldl update state) ∧ HeapRangeFold.Frame changed env finalEnv := by
  induction remaining generalizing position heap env state with
  | nil =>
      refine ⟨[], env, ?_, by simpa using safe, HeapRangeFold.Frame.refl changed env⟩
      apply BlockContinues.sequence_done (items := values.map encode)
        (by simp [sequenceValue, Mutable.sequence, found])
      · have missing : values[position]? = none := by rw [← List.head?_drop, suffix]; rfl
        simp [List.getElem?_map, missing]
      · simpa using (BlockContinues.nil (module := module) (env := env) (heap := heap) (globals := globals))
  | cons item rest ih =>
      have selected : values[position]? = some item := by rw [← List.head?_drop, suffix]; rfl
      have member : item ∈ values := List.mem_of_getElem? selected
      let bound := setLocal env name (encode item)
      have boundSafe := bindInvariant heap env state item safe member
      obtain ⟨extra, nextEnv, resumed, nextSafe, frame⟩ :=
        cell heap bound state item boundSafe member (by simp [bound, setLocal])
      have nextFound : (heap ++ extra)[address]? = some (.list (values.map encode)) :=
        HeapCollection.prefix_found _ _ _ _ found
      have nextSuffix : values.drop (position + 1) = rest := by
        simp only [List.drop_add_one_eq_tail_drop, suffix, List.tail_cons]
      obtain ⟨later, finalEnv, continued, finalSafe, finalFrame⟩ :=
        ih (position + 1) nextSuffix (heap ++ extra) nextEnv (update state item) nextSafe nextFound
      refine ⟨extra ++ later, finalEnv, ?_, ?_, ?_⟩
      · intro following anchors stack
        let next := Stmt.sequenceStepLoop name (.ref address) (position + 1) body []
        have traversal := (resumed next following anchors stack).trans (continued following anchors stack)
        have sequence : Mutable.sequence (sequenceValue (.ref address)) heap = some (.ok (values.map encode), heap) := by
          simp [sequenceValue, Mutable.sequence, found]
        have current : (values.map encode)[position]? = some (encode item) := by simp [List.getElem?_map, selected]
        have stepped : Reaches module anchors
            ⟨.block (.sequenceStepLoop name (.ref address) position body [] :: following) env, stack, heap⟩
            ⟨.block following finalEnv, stack, (heap ++ extra) ++ later⟩ globals :=
          .next (by simp [step, run, sequence, current, Calls.loopContinuation, control, next, bound]) traversal
        simpa only [List.append_assoc] using stepped
      · simpa only [List.append_assoc, List.foldl_cons] using finalSafe
      · exact (HeapRangeFold.Frame.updated changed env name (encode item) written).trans (frame.trans finalFrame)

end ProjectEulerSolutions.HeapLoop
