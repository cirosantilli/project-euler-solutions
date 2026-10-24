import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row12959_small : List Nat := []
theorem row12959_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12959 row12959_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12959_block_005 : List Nat := []
theorem row12959_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12959 row12959_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12959_block_006 : List Nat := []
theorem row12959_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12959 row12959_block_006 :=
  Budget.empty_row_before _ _ _ 12418 (fun q member => (Bounds.block_006 q member).2) (by decide)

def row12959_block_007 : List Nat := []
def plan12959_block_007 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, (.secondFactor 19), .second, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 17), (.firstFactor 37), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12959_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12959 row12959_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12959_block_007
  · decide
  · rfl

def row12959_block_008 : List Nat := []
theorem row12959_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12959 row12959_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12959_block_009 : List Nat := []
theorem row12959_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12959 row12959_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12959_block_010 : List Nat := []
theorem row12959_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12959 row12959_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12959_block_011 : List Nat := []
theorem row12959_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12959 row12959_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12959_block_012 : List Nat := []
theorem row12959_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12959 row12959_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12959_block_013 : List Nat := []
theorem row12959_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12959 row12959_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12959_large_left_left : List Nat := row12959_small ++ row12959_block_005
theorem row12959_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12959 row12959_large_left_left :=
  Budget.RowUpperBound.branch row12959_small_covers row12959_block_005_covers

def row12959_large_left_right_right : List Nat := row12959_block_007 ++ row12959_block_008
theorem row12959_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12959 row12959_large_left_right_right :=
  Budget.RowUpperBound.branch row12959_block_007_covers row12959_block_008_covers

def row12959_large_left_right : List Nat := row12959_block_006 ++ row12959_large_left_right_right
theorem row12959_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12959 row12959_large_left_right :=
  Budget.RowUpperBound.branch row12959_block_006_covers row12959_large_left_right_right_covers

def row12959_large_left : List Nat := row12959_large_left_left ++ row12959_large_left_right
theorem row12959_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12959 row12959_large_left :=
  Budget.RowUpperBound.branch row12959_large_left_left_covers row12959_large_left_right_covers

def row12959_large_right_left : List Nat := row12959_block_009 ++ row12959_block_010
theorem row12959_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12959 row12959_large_right_left :=
  Budget.RowUpperBound.branch row12959_block_009_covers row12959_block_010_covers

def row12959_large_right_right_right : List Nat := row12959_block_012 ++ row12959_block_013
theorem row12959_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12959 row12959_large_right_right_right :=
  Budget.RowUpperBound.branch row12959_block_012_covers row12959_block_013_covers

def row12959_large_right_right : List Nat := row12959_block_011 ++ row12959_large_right_right_right
theorem row12959_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12959 row12959_large_right_right :=
  Budget.RowUpperBound.branch row12959_block_011_covers row12959_large_right_right_right_covers

def row12959_large_right : List Nat := row12959_large_right_left ++ row12959_large_right_right
theorem row12959_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12959 row12959_large_right :=
  Budget.RowUpperBound.branch row12959_large_right_left_covers row12959_large_right_right_covers

def row12959_large : List Nat := row12959_large_left ++ row12959_large_right
theorem row12959_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12959 row12959_large :=
  Budget.RowUpperBound.branch row12959_large_left_covers row12959_large_right_covers

def row12959 : List Nat := row12959_large
theorem row12959_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12959 row12959 := row12959_large_covers

def row12967_small : List Nat := []
theorem row12967_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12967 row12967_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12967_block_005 : List Nat := []
theorem row12967_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12967 row12967_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12967_block_006 : List Nat := []
theorem row12967_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12967 row12967_block_006 :=
  Budget.empty_row_before _ _ _ 12418 (fun q member => (Bounds.block_006 q member).2) (by decide)

def row12967_block_007 : List Nat := [13003]
def plan12967_block_007 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.secondFactor 13), .first, .first, .possible, .first, .second, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12967_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12967 row12967_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12967_block_007
  · decide
  · rfl

def row12967_block_008 : List Nat := []
theorem row12967_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12967 row12967_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12967_block_009 : List Nat := []
theorem row12967_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12967 row12967_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12967_block_010 : List Nat := []
theorem row12967_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12967 row12967_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12967_block_011 : List Nat := []
theorem row12967_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12967 row12967_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12967_block_012 : List Nat := []
theorem row12967_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12967 row12967_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12967_block_013 : List Nat := []
theorem row12967_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12967 row12967_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12967_large_left_left : List Nat := row12967_small ++ row12967_block_005
theorem row12967_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12967 row12967_large_left_left :=
  Budget.RowUpperBound.branch row12967_small_covers row12967_block_005_covers

def row12967_large_left_right_right : List Nat := row12967_block_007 ++ row12967_block_008
theorem row12967_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12967 row12967_large_left_right_right :=
  Budget.RowUpperBound.branch row12967_block_007_covers row12967_block_008_covers

def row12967_large_left_right : List Nat := row12967_block_006 ++ row12967_large_left_right_right
theorem row12967_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12967 row12967_large_left_right :=
  Budget.RowUpperBound.branch row12967_block_006_covers row12967_large_left_right_right_covers

def row12967_large_left : List Nat := row12967_large_left_left ++ row12967_large_left_right
theorem row12967_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12967 row12967_large_left :=
  Budget.RowUpperBound.branch row12967_large_left_left_covers row12967_large_left_right_covers

def row12967_large_right_left : List Nat := row12967_block_009 ++ row12967_block_010
theorem row12967_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12967 row12967_large_right_left :=
  Budget.RowUpperBound.branch row12967_block_009_covers row12967_block_010_covers

def row12967_large_right_right_right : List Nat := row12967_block_012 ++ row12967_block_013
theorem row12967_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12967 row12967_large_right_right_right :=
  Budget.RowUpperBound.branch row12967_block_012_covers row12967_block_013_covers

def row12967_large_right_right : List Nat := row12967_block_011 ++ row12967_large_right_right_right
theorem row12967_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12967 row12967_large_right_right :=
  Budget.RowUpperBound.branch row12967_block_011_covers row12967_large_right_right_right_covers

def row12967_large_right : List Nat := row12967_large_right_left ++ row12967_large_right_right
theorem row12967_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12967 row12967_large_right :=
  Budget.RowUpperBound.branch row12967_large_right_left_covers row12967_large_right_right_covers

def row12967_large : List Nat := row12967_large_left ++ row12967_large_right
theorem row12967_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12967 row12967_large :=
  Budget.RowUpperBound.branch row12967_large_left_covers row12967_large_right_covers

def row12967 : List Nat := row12967_large
theorem row12967_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12967 row12967 := row12967_large_covers

def row12973_small : List Nat := []
theorem row12973_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12973 row12973_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12973_block_005 : List Nat := []
theorem row12973_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12973 row12973_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12973_block_006 : List Nat := []
theorem row12973_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12973 row12973_block_006 :=
  Budget.empty_row_before _ _ _ 12418 (fun q member => (Bounds.block_006 q member).2) (by decide)

def row12973_block_007 : List Nat := []
def plan12973_block_007 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .second, (.secondFactor 17), .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12973_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12973 row12973_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12973_block_007
  · decide
  · rfl

def row12973_block_008 : List Nat := []
theorem row12973_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12973 row12973_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12973_block_009 : List Nat := []
theorem row12973_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12973 row12973_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12973_block_010 : List Nat := []
theorem row12973_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12973 row12973_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12973_block_011 : List Nat := []
theorem row12973_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12973 row12973_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12973_block_012 : List Nat := []
theorem row12973_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12973 row12973_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12973_block_013 : List Nat := []
theorem row12973_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12973 row12973_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12973_large_left_left : List Nat := row12973_small ++ row12973_block_005
theorem row12973_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12973 row12973_large_left_left :=
  Budget.RowUpperBound.branch row12973_small_covers row12973_block_005_covers

def row12973_large_left_right_right : List Nat := row12973_block_007 ++ row12973_block_008
theorem row12973_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12973 row12973_large_left_right_right :=
  Budget.RowUpperBound.branch row12973_block_007_covers row12973_block_008_covers

def row12973_large_left_right : List Nat := row12973_block_006 ++ row12973_large_left_right_right
theorem row12973_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12973 row12973_large_left_right :=
  Budget.RowUpperBound.branch row12973_block_006_covers row12973_large_left_right_right_covers

def row12973_large_left : List Nat := row12973_large_left_left ++ row12973_large_left_right
theorem row12973_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12973 row12973_large_left :=
  Budget.RowUpperBound.branch row12973_large_left_left_covers row12973_large_left_right_covers

def row12973_large_right_left : List Nat := row12973_block_009 ++ row12973_block_010
theorem row12973_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12973 row12973_large_right_left :=
  Budget.RowUpperBound.branch row12973_block_009_covers row12973_block_010_covers

def row12973_large_right_right_right : List Nat := row12973_block_012 ++ row12973_block_013
theorem row12973_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12973 row12973_large_right_right_right :=
  Budget.RowUpperBound.branch row12973_block_012_covers row12973_block_013_covers

def row12973_large_right_right : List Nat := row12973_block_011 ++ row12973_large_right_right_right
theorem row12973_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12973 row12973_large_right_right :=
  Budget.RowUpperBound.branch row12973_block_011_covers row12973_large_right_right_right_covers

def row12973_large_right : List Nat := row12973_large_right_left ++ row12973_large_right_right
theorem row12973_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12973 row12973_large_right :=
  Budget.RowUpperBound.branch row12973_large_right_left_covers row12973_large_right_right_covers

def row12973_large : List Nat := row12973_large_left ++ row12973_large_right
theorem row12973_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12973 row12973_large :=
  Budget.RowUpperBound.branch row12973_large_left_covers row12973_large_right_covers

def row12973 : List Nat := row12973_large
theorem row12973_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12973 row12973 := row12973_large_covers

def row12979_small : List Nat := []
theorem row12979_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12979 row12979_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12979_block_005 : List Nat := []
theorem row12979_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12979 row12979_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12979_block_006 : List Nat := []
theorem row12979_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12979 row12979_block_006 :=
  Budget.empty_row_before _ _ _ 12418 (fun q member => (Bounds.block_006 q member).2) (by decide)

def row12979_block_007 : List Nat := [13033]
def plan12979_block_007 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, (.secondFactor 7), .first, (.firstFactor 19), .possible, .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12979_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12979 row12979_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12979_block_007
  · decide
  · rfl

def row12979_block_008 : List Nat := []
theorem row12979_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12979 row12979_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12979_block_009 : List Nat := []
theorem row12979_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12979 row12979_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12979_block_010 : List Nat := []
theorem row12979_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12979 row12979_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12979_block_011 : List Nat := []
theorem row12979_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12979 row12979_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12979_block_012 : List Nat := []
theorem row12979_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12979 row12979_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12979_block_013 : List Nat := []
theorem row12979_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12979 row12979_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12979_large_left_left : List Nat := row12979_small ++ row12979_block_005
theorem row12979_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12979 row12979_large_left_left :=
  Budget.RowUpperBound.branch row12979_small_covers row12979_block_005_covers

def row12979_large_left_right_right : List Nat := row12979_block_007 ++ row12979_block_008
theorem row12979_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12979 row12979_large_left_right_right :=
  Budget.RowUpperBound.branch row12979_block_007_covers row12979_block_008_covers

def row12979_large_left_right : List Nat := row12979_block_006 ++ row12979_large_left_right_right
theorem row12979_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12979 row12979_large_left_right :=
  Budget.RowUpperBound.branch row12979_block_006_covers row12979_large_left_right_right_covers

def row12979_large_left : List Nat := row12979_large_left_left ++ row12979_large_left_right
theorem row12979_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12979 row12979_large_left :=
  Budget.RowUpperBound.branch row12979_large_left_left_covers row12979_large_left_right_covers

def row12979_large_right_left : List Nat := row12979_block_009 ++ row12979_block_010
theorem row12979_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12979 row12979_large_right_left :=
  Budget.RowUpperBound.branch row12979_block_009_covers row12979_block_010_covers

def row12979_large_right_right_right : List Nat := row12979_block_012 ++ row12979_block_013
theorem row12979_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12979 row12979_large_right_right_right :=
  Budget.RowUpperBound.branch row12979_block_012_covers row12979_block_013_covers

def row12979_large_right_right : List Nat := row12979_block_011 ++ row12979_large_right_right_right
theorem row12979_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12979 row12979_large_right_right :=
  Budget.RowUpperBound.branch row12979_block_011_covers row12979_large_right_right_right_covers

def row12979_large_right : List Nat := row12979_large_right_left ++ row12979_large_right_right
theorem row12979_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12979 row12979_large_right :=
  Budget.RowUpperBound.branch row12979_large_right_left_covers row12979_large_right_right_covers

def row12979_large : List Nat := row12979_large_left ++ row12979_large_right
theorem row12979_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12979 row12979_large :=
  Budget.RowUpperBound.branch row12979_large_left_covers row12979_large_right_covers

def row12979 : List Nat := row12979_large
theorem row12979_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12979 row12979 := row12979_large_covers

end ProjectEulerSolutions.PrimePairLarge
