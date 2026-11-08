import ProjectEulerSolutions.Lib.MutableIntegerMatrixAccess

/-! The distinct rows and temporary singleton lists allocated by `[[0] * width for _ in range(height)]`. -/
namespace ProjectEulerSolutions.ZeroMatrixAllocation
open FormalSW.PyAst HeapCalls
set_option Elab.async false

def block (width : Nat) : Mutable.Heap :=
  [.list [.scalar (.int 0)], .list (List.replicate width (.scalar (.int 0)))]

def objects (width height : Nat) : Mutable.Heap := (List.range height).flatMap fun _ => block width

def address (base : Mutable.Heap) (row : Nat) : Nat := base.length + 2 + 2 * row

def references (base : Mutable.Heap) (height : Nat) : List Mutable.Value :=
  (List.range height).map fun row => .ref (address base row)

def working (base : Mutable.Heap) (width height : Nat) : Mutable.Heap :=
  HeapCollection.working base (references base height) (objects width height)

theorem objects_length (width height : Nat) : (objects width height).length = 2 * height := by
  simp [objects, block, List.length_flatMap, Nat.mul_comm]

theorem objects_succ (width height : Nat) : objects width (height + 1) = objects width height ++ block width := by
  simp [objects, List.range_succ]

theorem working_length (base : Mutable.Heap) (width height : Nat) :
    (working base width height).length = base.length + 1 + 2 * height := by
  simp only [working, HeapCollection.working, List.length_append, List.length_cons, objects_length]
  omega

theorem objects_row (width height index : Nat) (bound : index < height) :
    (objects width height)[2 * index + 1]? = some (.list (List.replicate width (.scalar (.int 0)))) := by
  induction height with
  | zero => omega
  | succ height ih =>
      rw [objects_succ]
      by_cases previous : index < height
      · rw [List.getElem?_append_left (by rw [objects_length]; omega)]
        exact ih previous
      · have same : index = height := by omega
        subst index
        rw [List.getElem?_append_right (by rw [objects_length]; omega), objects_length]
        simp [block]

theorem row_found (base : Mutable.Heap) (width height index : Nat) (bound : index < height) :
    (working base width height)[address base index]? = some (.list (List.replicate width (.scalar (.int 0)))) := by
  unfold working HeapCollection.working address
  rw [List.getElem?_append_right (by omega)]
  have offset : base.length + 2 + 2 * index - base.length = (2 * index + 1) + 1 := by omega
  rw [offset, List.getElem?_cons_succ]
  exact objects_row width height index bound

theorem view (base : Mutable.Heap) (width height : Nat) :
    MutableIntegerMatrix.View (fun (_ : Fin height) (_ : Fin width) => 0)
      (working base width height) base.length (fun row => address base row.val) := by
  refine ⟨?_, ?_, ?_, ?_⟩
  · have same : references base height = List.ofFn (fun row : Fin height => Mutable.Value.ref (address base row.val)) := by
      apply List.ext_getElem
      · simp [references]
      · intro index left right
        simp [references]
    simpa only [working, same] using HeapCollection.found base (references base height) (objects width height)
  · intro row
    simpa using row_found base width height row.val row.isLt
  · intro first second same
    apply Fin.ext
    change base.length + 2 + 2 * first.val = base.length + 2 + 2 * second.val at same
    omega
  · intro row
    unfold address
    omega

def rowExpression (width : Nat) : Expr := .binOp (.list [.constant (.int 0)]) .mult (.constant (.int width))

theorem row_result (module : Module) (width : Nat) (env : HeapCalls.Env) (heap : Mutable.Heap) (globals : Mutable.Globals) :
    Evaluates module env (rowExpression width) heap (.ref (heap.length + 1)) (heap ++ block width) globals := by
  apply Evaluates.binary (Evaluates.list (.cons Evaluates.constant .nil)) Evaluates.constant
  simp [Mutable.binary, show (BinOp.mult == BinOp.mult) = true from rfl,
    Mutable.array, Mutable.allocate, block, Except.map]

private theorem loop (module : Module) (name : String) (base : Mutable.Heap) (width first count stop : Nat)
    (endpoint : first + count = stop) (env : HeapCalls.Env) (globals : Mutable.Globals) :
    ∃ finalEnv, BlockContinues module env
      [.rangeStepLoop name first stop 1 [.comprehensionAppend base.length (rowExpression width)] []]
      (working base width first) finalEnv (working base width stop) globals := by
  induction count generalizing first env with
  | zero =>
      have same : first = stop := by omega
      subst first
      exact ⟨env, BlockContinues.range_done (by simp [Calls.rangeContinues]) .nil⟩
  | succ count ih =>
      let entered := setLocal env name (.scalar (.int first))
      have allocated := row_result module width entered (working base width first) globals
      have emitted := HeapCollection.emit base (references base first) (objects width first) (block width)
        (.ref ((working base width first).length + 1)) allocated
      have allocatedAddress : (working base width first).length + 1 = address base first := by
        rw [working_length]
        unfold address
        omega
      rw [allocatedAddress] at emitted
      have stored : BlockContinues module entered [.comprehensionAppend base.length (rowExpression width)]
          (working base width first) entered (working base width (first + 1)) globals := by
        simpa only [working, references, objects_succ, List.range_succ, List.map_append, List.map_singleton] using emitted
      obtain ⟨finalEnv, continued⟩ := ih (first + 1) (by omega) entered
      refine ⟨finalEnv, ?_⟩
      apply BlockContinues.range_step (by simp [Calls.rangeContinues]; omega)
      exact stored.append continued

/-- Each multiplication returns a fresh independent row; singleton temporaries remain private. -/
theorem result (module : Module) (name : String) (base : Mutable.Heap) (width height : Nat)
    (env : HeapCalls.Env) (globals : Mutable.Globals) :
    Evaluates module env (.listComp (rowExpression width) [(name, .call (.name "range") [.constant (.int height)], [])])
      base (.ref base.length) (working base width height) globals := by
  obtain ⟨finalEnv, executed⟩ := loop module name base width 0 height height (by omega)
    (comprehensionEnv env [(name, .call (.name "range") [.constant (.int height)], [])]) globals
  apply Evaluates.list_comp_range (.cons Evaluates.constant .nil) rfl
  exact executed

end ProjectEulerSolutions.ZeroMatrixAllocation
