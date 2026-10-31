import solvers.«90.py.lean.d».Enumeration

/-! The expanded-mask comprehension preserves its source list and every caller object. -/
namespace solvers.«90».expansion
open FormalSW.PyAst HeapCalls
open ProjectEulerSolutions DigitDice
set_option Elab.async false

def element : Expr := .call (.name "expand_6_9") [.name "m"]
def body (address : Nat) : List Stmt := [.comprehensionAppend address element]
def expression : Expr := .listComp element [("m", .name "masks", [])]

private theorem loop (base : Mutable.Heap) (address : Nat) (values remaining output : List Nat)
    (position : Nat) (env : HeapCalls.Env) (globals : Mutable.Globals)
    (found : base[address]? = some (.list (values.map number)))
    (suffix : values.drop position = remaining) :
    ∃ finalEnv, BlockContinues program env
      [.sequenceStepLoop "m" (.ref address) position (body base.length) []]
      (masks.working base output []) finalEnv
      (masks.working base (output ++ remaining.map rotateMask) []) globals := by
  induction remaining generalizing position output env with
  | nil =>
      refine ⟨env, ?_⟩
      simp only [List.map_nil, List.append_nil]
      apply BlockContinues.sequence_done (items := values.map number)
      · have kept := HeapCollection.prefix_found base [.list (output.map number)] address _ found
        simpa only [sequenceValue, Mutable.sequence, kept] using
          (show Mutable.sequence (.ref address) (masks.working base output []) =
            some (.ok (values.map number), masks.working base output []) from by
              simp [Mutable.sequence, masks.working, HeapCollection.working, kept])
      · have missing : values[position]? = none :=
          List.getElem?_eq_none_iff.mpr (List.drop_eq_nil_iff.mp suffix)
        simp [missing]
      · exact BlockContinues.nil
  | cons value rest ih =>
      have item : values[position]? = some value := by
        simpa [suffix] using (List.head?_drop (l := values) (i := position)).symm
      have tail : values.drop (position + 1) = rest := by
        simp [List.drop_add_one_eq_tail_drop, suffix]
      let bound := setLocal env "m" (number value)
      have evaluated : Evaluates program bound element (masks.working base output [])
          (number (rotateMask value)) (masks.working base output []) globals :=
        Evaluates.call (.cons (Evaluates.lookup rfl) .nil)
          (expand_6_9.result value (masks.working base output []) globals)
      have emitted : BlockContinues program bound (body base.length)
          (masks.working base output []) bound (masks.working base (output ++ [rotateMask value]) []) globals := by
        have emission := HeapCollection.emit (globals := globals) base (output.map number) [] []
          (number (rotateMask value)) (by simpa only [List.append_nil] using evaluated)
        simpa only [body, masks.working, List.append_nil, List.nil_append,
          List.map_append, List.map_cons, List.map_nil] using emission
      obtain ⟨finalEnv, continued⟩ := ih (output ++ [rotateMask value]) (position + 1) bound tail
      refine ⟨finalEnv, ?_⟩
      apply BlockContinues.sequence_step (items := values.map number) (value := number value)
      · have kept := HeapCollection.prefix_found base [.list (output.map number)] address _ found
        simp [sequenceValue, Mutable.sequence, masks.working, HeapCollection.working, kept]
      · simp [item]
      · simpa only [List.map_cons, List.append_assoc, List.singleton_append] using emitted.append continued

/-- The actual source comprehension maps rotation over all masks, retaining their multiplicity. -/
theorem result (values : List Nat) (address : Nat) (heap : Mutable.Heap)
    (env : HeapCalls.Env) (globals : Mutable.Globals)
    (found : heap[address]? = some (.list (values.map number)))
    (getMasks : env "masks" = some (.ref address)) :
    Evaluates program env expression heap (.ref heap.length)
      (heap ++ [.list ((values.map rotateMask).map number)]) globals := by
  obtain ⟨finalEnv, executed⟩ := loop heap address values values [] 0
    (comprehensionEnv env [("m", .name "masks", [])]) globals found rfl
  apply Evaluates.list_comp_sequence (by intros; simp) (Evaluates.lookup getMasks)
    (items := values.map number)
  · simp [Mutable.sequence, found]
  · exact executed

end solvers.«90».expansion
