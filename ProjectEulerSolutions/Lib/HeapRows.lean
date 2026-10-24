import ProjectEulerSolutions.Lib.HeapCollection

/-! Exact shared-heap allocation of the independent rows in `[[] for _ in range(n)]`. -/
namespace ProjectEulerSolutions.HeapRows
open FormalSW.PyAst FormalSW.PyAst.HeapCalls

set_option Elab.async false

def addresses (base count : Nat) : List Nat := (List.range count).map (fun offset => base + 1 + offset)

def empty (base : Mutable.Heap) (count : Nat) : Mutable.Heap :=
  HeapCollection.working base ((addresses base.length count).map Mutable.Value.ref) (List.replicate count (.list []))

theorem parent_found (base : Mutable.Heap) (count : Nat) :
    (empty base count)[base.length]? = some (.list ((addresses base.length count).map Mutable.Value.ref)) :=
  HeapCollection.found _ _ _

theorem prefix_found (base : Mutable.Heap) (count address : Nat) (object : Mutable.Object)
    (found : base[address]? = some object) : (empty base count)[address]? = some object :=
  HeapCollection.prefix_found base _ address object found

theorem row_found (base : Mutable.Heap) (count index : Nat) (inside : index < count) :
    (empty base count)[base.length + 1 + index]? = some (.list []) := by
  unfold empty HeapCollection.working
  rw [List.getElem?_append_right (by omega)]
  have offset : base.length + 1 + index - base.length = index + 1 := by omega
  rw [offset, List.getElem?_cons_succ]
  exact List.getElem?_replicate_of_lt inside

private theorem loop (program : Module) (name : String) (base : Mutable.Heap) (first remaining stop : Nat)
    (endpoint : first + remaining = stop) (env : HeapCalls.Env) :
    ∃ finalEnv, BlockContinues program env
      [.rangeStepLoop name first stop 1 [.comprehensionAppend base.length (.list [])] []]
      (empty base first) finalEnv (empty base stop) := by
  induction remaining generalizing first env with
  | zero =>
      have equal : first = stop := by omega
      refine ⟨env, ?_⟩
      rw [← equal]
      exact BlockContinues.range_done (by simp [Calls.rangeContinues]) .nil
  | succ remaining ih =>
      let bound := setLocal env name (.scalar (.int first))
      have allocated : Evaluates program bound (.list []) (empty base first)
          (.ref (base.length + 1 + first)) (empty base first ++ [.list []]) := by
        simpa only [empty, HeapCollection.working, List.length_append, List.length_cons, List.length_replicate,
          Nat.add_assoc, Nat.add_comm 1 first] using (Evaluates.list (module := program) (env := bound) (heap := empty base first) .nil)
      have emitted := HeapCollection.emit base ((addresses base.length first).map Mutable.Value.ref)
        (List.replicate first (.list [])) [.list []] (.ref (base.length + 1 + first)) allocated
      have stored : BlockContinues program bound [.comprehensionAppend base.length (.list [])]
          (empty base first) bound (empty base (first + 1)) := by
        simpa only [empty, addresses, List.range_succ, List.map_append, List.map_singleton, ← List.replicate_append_replicate,
          List.replicate_one] using emitted
      obtain ⟨finalEnv, executed⟩ := ih (first + 1) (by omega) bound
      refine ⟨finalEnv, ?_⟩
      apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
      exact stored.append executed

/-- A range comprehension allocates one fresh empty list per iteration and stores its actual address. -/
theorem allocate (program : Module) (name : String) (base : Mutable.Heap) (count : Nat) (env : HeapCalls.Env) :
    Evaluates program env (.listComp (.list []) [(name, .call (.name "range") [.constant (.int count)], [])])
      base (.ref base.length) (empty base count) := by
  obtain ⟨finalEnv, executed⟩ := loop program name base 0 count count (by omega)
    (comprehensionEnv env [(name, .call (.name "range") [.constant (.int count)], [])])
  apply Evaluates.list_comp_range (.cons Evaluates.constant .nil) rfl
  exact executed

end ProjectEulerSolutions.HeapRows
