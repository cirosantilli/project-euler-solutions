import solvers.«83.py.lean.d».Allocation

namespace solvers.«83»
open FormalSW.PyAst HeapCalls
open ProjectEulerSolutions.RectangularMatrixCalls (MatrixRows)
open ProjectEulerSolutions.TriangleBufferCalls (number encode)
set_option Elab.async false

abbrev queueAnnotation : Expr := .subscript (.name "list")
  (.subscript (.name "tuple") (.tuple [.name "int", .name "int"]))

abbrev setupCode : List Stmt :=
  [.assign [.name "infinity"] (.call (.name "big_val") [.name "matrix"]),
   .assign [.name "dist"] distanceExpression,
   .assign [.name "start"] (.constant (.int 0)),
   .assign [.name "goal"] (.binOp (.name "total") .sub (.constant (.int 1))),
   .assign [.subscript (.name "dist") (.name "start")] firstCell,
   .annAssign (.name "heap") queueAnnotation (some queueExpression),
   .assign [.name "visited"] visitedExpression]

/-- This is the allocation prefix of the imported solver, immediately after `total = n * m`. -/
theorem setup_source : ∃ args body, findFunctionIn program.body "solve" = some (args, body) ∧
    ((executableBody body).drop 6).take 7 = setupCode := by
  exact ⟨_, _, rfl, rfl⟩

def initialDistances (size : Nat) (bound distance : Int) : List Int :=
  (List.replicate size bound).set 0 distance

def initialHeap (heap : Mutable.Heap) (size : Nat) (bound distance : Int) : Mutable.Heap :=
  visitedHeap (queueHeap (distanceHeap heap bound (initialDistances size bound distance)) distance) size

def initialEnv (env : HeapCalls.Env) (heap : Mutable.Heap) (size : Nat) (bound : Int) : HeapCalls.Env :=
  setLocal (setLocal (setLocal (setLocal (setLocal (setLocal env "infinity" (number bound))
    "dist" (.ref (heap.length + 1))) "start" (number 0)) "goal" (number ((size : Int) - 1)))
    "heap" (.ref (heap.length + 3))) "visited" (.ref (heap.length + 5))

theorem setup_result (heap : Mutable.Heap) (env : HeapCalls.Env) (address size : Nat)
    (first : Nat × List Int) (rest : MatrixRows)
    (found : heap[address]? = some (.list ((first :: rest).map (fun pair => .ref pair.1))))
    (reads : ∀ pair ∈ first :: rest, heap[pair.1]? = some (.list (encode pair.2)))
    (positive : 0 < first.2.length) (sizePositive : 0 < size)
    (localMatrix : env "matrix" = some (.ref address)) (localSize : env "total" = some (number size)) :
    let bound := ProjectEulerStatements.P83.bigVal ((first :: rest).map Prod.snd)
    BlockContinues program env setupCode heap (initialEnv env heap size bound)
      (initialHeap heap size bound first.2[0]) := by
  intro bound
  let bounded := setLocal env "infinity" (number bound)
  let buffered := setLocal bounded "dist" (.ref (heap.length + 1))
  let started := setLocal buffered "start" (number 0)
  let targeted := setLocal started "goal" (number ((size : Int) - 1))
  let distances := distanceHeap heap bound (initialDistances size bound first.2[0])
  let queued := setLocal targeted "heap" (.ref (distances.length + 1))
  have kept (values : List Int) (target : Nat) (object : Mutable.Object)
      (read : heap[target]? = some object) : (distanceHeap heap bound values)[target]? = some object := by
    rw [distanceHeap, List.getElem?_append_left (List.getElem?_eq_some_iff.mp read).1]
    exact read
  have boundCall : Evaluates program env (.call (.name "big_val") [.name "matrix"])
      heap (number bound) heap :=
    Evaluates.call (.cons (Evaluates.lookup localMatrix) .nil) (big_val.result heap address (first :: rest) found reads)
  have allocated := allocate_distances heap bounded size bound
    (by simpa [bounded, setLocal] using localSize) rfl
  have goal : Evaluates program started (.binOp (.name "total") .sub (.constant (.int 1)))
      (distanceHeap heap bound (List.replicate size bound)) (number ((size : Int) - 1))
      (distanceHeap heap bound (List.replicate size bound)) :=
    Evaluates.binary (Evaluates.lookup (by simpa [started, buffered, bounded, setLocal] using localSize))
      Evaluates.constant rfl
  have seeded := set_initial_distance heap targeted size bound first.2[0] sizePositive
    (by simp [targeted, started, buffered, setLocal]) (by simp [targeted, started, setLocal])
    (first_cell _ targeted address first rest (kept _ _ _ found)
      (kept _ _ _ (reads first (by simp))) positive
      (by simpa [targeted, started, buffered, bounded, setLocal] using localMatrix))
  have queue := allocate_queue distances targeted first.2[0] (by simp [targeted, started, setLocal])
    (first_cell _ targeted address first rest (kept _ _ _ found)
      (kept _ _ _ (reads first (by simp))) positive
      (by simpa [targeted, started, buffered, bounded, setLocal] using localMatrix))
  have visited := allocate_visited (queueHeap distances first.2[0]) queued size
    (by simpa [queued, targeted, started, buffered, bounded, setLocal] using localSize)
  have ran := (BlockContinues.assign (name := "infinity") boundCall).append
    ((BlockContinues.assign (name := "dist") allocated).append
      ((BlockContinues.assign (name := "start")
        (Evaluates.constant (module := program) (env := buffered)
          (heap := distanceHeap heap bound (List.replicate size bound)) (value := .int 0))).append
        ((BlockContinues.assign (name := "goal") goal).append
          (seeded.append ((BlockContinues.annAssign (name := "heap") (annotation := queueAnnotation) queue).append
            (BlockContinues.assign (name := "visited") visited))))))
  simpa only [setupCode, initialEnv, initialHeap, initialDistances, distances, queued, targeted,
    started, buffered, bounded, distanceHeap, queueHeap, List.length_append, List.length_cons,
    List.length_nil, Nat.add_zero, Nat.add_assoc, List.cons_append, List.nil_append] using ran

end solvers.«83»
