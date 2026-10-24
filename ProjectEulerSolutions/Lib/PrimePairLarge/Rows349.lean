import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row11633_small : List Nat := []
theorem row11633_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11633 row11633_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11633_block_005 : List Nat := []
theorem row11633_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11633 row11633_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11633_block_006 : List Nat := [11927, 12251]
def plan11633_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 37), .first, (.secondFactor 37), (.firstFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 11), (.firstFactor 13), .first, .first, .possible, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 13), (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 19), .first, (.secondFactor 7), (.firstFactor 29), (.firstFactor 7), (.secondFactor 13), .first, (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 23), .first, .possible, .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 17), .first, .first, .first, (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 23)]
set_option maxRecDepth 2048 in
theorem row11633_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11633 row11633_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11633_block_006
  · decide
  · rfl

def row11633_block_007 : List Nat := [12713]
def plan11633_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 7), .first, .first, .second, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 17), .first, .first, .first, (.firstFactor 13), (.secondFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, .first, (.firstFactor 29), (.firstFactor 7), (.firstFactor 23), (.secondFactor 7), (.firstFactor 37), (.firstFactor 7), .first, .first, .possible, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 19), .first, .second, (.firstFactor 11), .first, .first, .first, .first, .first, (.firstFactor 37), (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 7), (.firstFactor 11), .second, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, (.firstFactor 31), (.firstFactor 17), .first, .first, .first, .first, (.secondFactor 23), (.firstFactor 7), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 29), .first, .first, .first, .second, (.firstFactor 11), .first, (.firstFactor 13), (.secondFactor 13), .first, (.secondFactor 31), .first, .first, .first, .first, (.secondFactor 17), .first, (.secondFactor 7), (.firstFactor 29), .first, .second, .first, (.secondFactor 13), .first, .first, .first, .second, .first, (.firstFactor 17), (.secondFactor 7), (.firstFactor 13), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 19), .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 11), .first, .first, (.secondFactor 19), .first, (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 17), .first, .first, (.secondFactor 7), (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.secondFactor 37), .first, .second, .first, .first, (.firstFactor 17), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 31), .second, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .second, .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 31), (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 11), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.firstFactor 19), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11633_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11633 row11633_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11633_block_007
  · decide
  · rfl

def row11633_block_008 : List Nat := []
theorem row11633_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11633 row11633_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row11633_block_009 : List Nat := []
theorem row11633_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11633 row11633_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11633_block_010 : List Nat := []
theorem row11633_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11633 row11633_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11633_block_011 : List Nat := []
theorem row11633_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11633 row11633_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11633_block_012 : List Nat := []
theorem row11633_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11633 row11633_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11633_block_013 : List Nat := []
theorem row11633_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11633 row11633_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11633_large_left_left : List Nat := row11633_small ++ row11633_block_005
theorem row11633_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11633 row11633_large_left_left :=
  Budget.RowUpperBound.branch row11633_small_covers row11633_block_005_covers

def row11633_large_left_right_right : List Nat := row11633_block_007 ++ row11633_block_008
theorem row11633_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11633 row11633_large_left_right_right :=
  Budget.RowUpperBound.branch row11633_block_007_covers row11633_block_008_covers

def row11633_large_left_right : List Nat := row11633_block_006 ++ row11633_large_left_right_right
theorem row11633_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11633 row11633_large_left_right :=
  Budget.RowUpperBound.branch row11633_block_006_covers row11633_large_left_right_right_covers

def row11633_large_left : List Nat := row11633_large_left_left ++ row11633_large_left_right
theorem row11633_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11633 row11633_large_left :=
  Budget.RowUpperBound.branch row11633_large_left_left_covers row11633_large_left_right_covers

def row11633_large_right_left : List Nat := row11633_block_009 ++ row11633_block_010
theorem row11633_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11633 row11633_large_right_left :=
  Budget.RowUpperBound.branch row11633_block_009_covers row11633_block_010_covers

def row11633_large_right_right_right : List Nat := row11633_block_012 ++ row11633_block_013
theorem row11633_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11633 row11633_large_right_right_right :=
  Budget.RowUpperBound.branch row11633_block_012_covers row11633_block_013_covers

def row11633_large_right_right : List Nat := row11633_block_011 ++ row11633_large_right_right_right
theorem row11633_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11633 row11633_large_right_right :=
  Budget.RowUpperBound.branch row11633_block_011_covers row11633_large_right_right_right_covers

def row11633_large_right : List Nat := row11633_large_right_left ++ row11633_large_right_right
theorem row11633_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11633 row11633_large_right :=
  Budget.RowUpperBound.branch row11633_large_right_left_covers row11633_large_right_right_covers

def row11633_large : List Nat := row11633_large_left ++ row11633_large_right
theorem row11633_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11633 row11633_large :=
  Budget.RowUpperBound.branch row11633_large_left_covers row11633_large_right_covers

def row11633 : List Nat := row11633_large
theorem row11633_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11633 row11633 := row11633_large_covers

def row11657_small : List Nat := []
theorem row11657_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11657 row11657_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11657_block_005 : List Nat := []
theorem row11657_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11657 row11657_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11657_block_006 : List Nat := [11699]
def plan11657_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, .first, .possible, .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 23), (.firstFactor 11), .first, (.firstFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, .first, (.secondFactor 23), .first, (.firstFactor 7), .first, (.secondFactor 13), .first, (.firstFactor 29), .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 19), .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 7), .first, .second, (.firstFactor 11), (.secondFactor 13), (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .second, .first, (.secondFactor 37), (.secondFactor 19), .first, (.firstFactor 11), .first, (.firstFactor 19), (.firstFactor 37), .first, (.secondFactor 23), .first, .first, (.secondFactor 17), .first, .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), .first, .first]
set_option maxRecDepth 2048 in
theorem row11657_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11657 row11657_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11657_block_006
  · decide
  · rfl

def row11657_block_007 : List Nat := []
def plan11657_block_007 : List Budget.Choice := [.first, .first, (.firstFactor 31), .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 17), .second, (.secondFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 29), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.firstFactor 31), (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 37), (.firstFactor 13), .first, (.firstFactor 11), .second, .first, (.secondFactor 7), .first, (.secondFactor 31), .first, .first, .first, .first, .first, (.firstFactor 17), .first, (.secondFactor 7), .first, .first, (.secondFactor 17), (.firstFactor 11), (.secondFactor 7), .first, .first, .first, (.firstFactor 17), (.firstFactor 11), .second, .second, .first, (.secondFactor 19), .first, (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, .first, (.secondFactor 13), .first, .second, (.secondFactor 37), .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .second, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 31), .first, (.firstFactor 7), .first, .first, (.firstFactor 17), (.firstFactor 7), .first, .first, (.secondFactor 17), .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 31), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 17), (.firstFactor 13), .first, (.secondFactor 7), (.secondFactor 13), .first, (.secondFactor 23), .first, (.firstFactor 7), .first, (.firstFactor 29), (.firstFactor 19), .first, .second, .first, (.secondFactor 31), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 17), .first, (.secondFactor 19), .first, (.firstFactor 7), .first, .second, (.firstFactor 7), .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 17), .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11657_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11657 row11657_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11657_block_007
  · decide
  · rfl

def row11657_block_008 : List Nat := []
theorem row11657_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11657 row11657_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row11657_block_009 : List Nat := []
theorem row11657_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11657 row11657_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11657_block_010 : List Nat := []
theorem row11657_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11657 row11657_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11657_block_011 : List Nat := []
theorem row11657_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11657 row11657_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11657_block_012 : List Nat := []
theorem row11657_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11657 row11657_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11657_block_013 : List Nat := []
theorem row11657_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11657 row11657_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11657_large_left_left : List Nat := row11657_small ++ row11657_block_005
theorem row11657_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11657 row11657_large_left_left :=
  Budget.RowUpperBound.branch row11657_small_covers row11657_block_005_covers

def row11657_large_left_right_right : List Nat := row11657_block_007 ++ row11657_block_008
theorem row11657_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11657 row11657_large_left_right_right :=
  Budget.RowUpperBound.branch row11657_block_007_covers row11657_block_008_covers

def row11657_large_left_right : List Nat := row11657_block_006 ++ row11657_large_left_right_right
theorem row11657_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11657 row11657_large_left_right :=
  Budget.RowUpperBound.branch row11657_block_006_covers row11657_large_left_right_right_covers

def row11657_large_left : List Nat := row11657_large_left_left ++ row11657_large_left_right
theorem row11657_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11657 row11657_large_left :=
  Budget.RowUpperBound.branch row11657_large_left_left_covers row11657_large_left_right_covers

def row11657_large_right_left : List Nat := row11657_block_009 ++ row11657_block_010
theorem row11657_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11657 row11657_large_right_left :=
  Budget.RowUpperBound.branch row11657_block_009_covers row11657_block_010_covers

def row11657_large_right_right_right : List Nat := row11657_block_012 ++ row11657_block_013
theorem row11657_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11657 row11657_large_right_right_right :=
  Budget.RowUpperBound.branch row11657_block_012_covers row11657_block_013_covers

def row11657_large_right_right : List Nat := row11657_block_011 ++ row11657_large_right_right_right
theorem row11657_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11657 row11657_large_right_right :=
  Budget.RowUpperBound.branch row11657_block_011_covers row11657_large_right_right_right_covers

def row11657_large_right : List Nat := row11657_large_right_left ++ row11657_large_right_right
theorem row11657_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11657 row11657_large_right :=
  Budget.RowUpperBound.branch row11657_large_right_left_covers row11657_large_right_right_covers

def row11657_large : List Nat := row11657_large_left ++ row11657_large_right
theorem row11657_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11657 row11657_large :=
  Budget.RowUpperBound.branch row11657_large_left_covers row11657_large_right_covers

def row11657 : List Nat := row11657_large
theorem row11657_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11657 row11657 := row11657_large_covers

def row11677_small : List Nat := []
theorem row11677_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11677 row11677_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11677_block_005 : List Nat := []
theorem row11677_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11677 row11677_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11677_block_006 : List Nat := [12097, 12379]
def plan11677_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.firstFactor 31), .first, (.firstFactor 13), .first, (.secondFactor 17), (.secondFactor 37), (.firstFactor 11), .first, (.firstFactor 13), .first, .first, .first, .first, .first, (.secondFactor 17), (.firstFactor 17), .first, .first, (.firstFactor 7), (.firstFactor 23), .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 37), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .second, (.firstFactor 7), .first, (.firstFactor 11), .possible, .first, .first, (.secondFactor 31), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .second, .first, .first, (.firstFactor 19), .first, .first, .first, .first, (.firstFactor 23), .first, (.firstFactor 29), (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .possible, .first, .first, .first, .first]
set_option maxRecDepth 2048 in
theorem row11677_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11677 row11677_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11677_block_006
  · decide
  · rfl

def row11677_block_007 : List Nat := [12757, 13441, 13567, 13669, 13999, 14143]
def plan11677_block_007 : List Budget.Choice := [.first, (.firstFactor 31), .first, .first, (.secondFactor 7), .first, .first, .second, .first, .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .second, (.firstFactor 19), .first, (.firstFactor 11), .first, .first, (.firstFactor 31), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 7), (.secondFactor 17), .first, .possible, (.firstFactor 7), .first, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 23), (.secondFactor 17), (.secondFactor 13), (.firstFactor 7), .first, .first, (.secondFactor 23), .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 23), (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 13), .second, .first, .first, .first, (.firstFactor 11), .second, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 29), .first, (.secondFactor 7), (.firstFactor 31), (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 17), .first, (.firstFactor 7), .first, (.secondFactor 7), (.firstFactor 7), .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, (.firstFactor 19), .first, (.secondFactor 13), (.secondFactor 31), (.firstFactor 13), .first, .possible, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .second, .first, (.secondFactor 19), .first, .possible, .first, (.firstFactor 11), (.secondFactor 23), .first, .first, .first, (.secondFactor 7), .first, .possible, .first, (.firstFactor 29), (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 13), .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 23), .first, (.secondFactor 19), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 23), (.firstFactor 13), .first, .first, .possible, .first, (.secondFactor 7), (.firstFactor 29), .first, .first, .first, (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .possible, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 13), .first, (.secondFactor 7), .first, .first, (.firstFactor 11), .first, (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 31), (.firstFactor 23), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11677_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11677 row11677_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11677_block_007
  · decide
  · rfl

def row11677_block_008 : List Nat := []
theorem row11677_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11677 row11677_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row11677_block_009 : List Nat := []
theorem row11677_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11677 row11677_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11677_block_010 : List Nat := []
theorem row11677_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11677 row11677_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11677_block_011 : List Nat := []
theorem row11677_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11677 row11677_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11677_block_012 : List Nat := []
theorem row11677_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11677 row11677_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11677_block_013 : List Nat := []
theorem row11677_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11677 row11677_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11677_large_left_left : List Nat := row11677_small ++ row11677_block_005
theorem row11677_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11677 row11677_large_left_left :=
  Budget.RowUpperBound.branch row11677_small_covers row11677_block_005_covers

def row11677_large_left_right_right : List Nat := row11677_block_007 ++ row11677_block_008
theorem row11677_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11677 row11677_large_left_right_right :=
  Budget.RowUpperBound.branch row11677_block_007_covers row11677_block_008_covers

def row11677_large_left_right : List Nat := row11677_block_006 ++ row11677_large_left_right_right
theorem row11677_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11677 row11677_large_left_right :=
  Budget.RowUpperBound.branch row11677_block_006_covers row11677_large_left_right_right_covers

def row11677_large_left : List Nat := row11677_large_left_left ++ row11677_large_left_right
theorem row11677_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11677 row11677_large_left :=
  Budget.RowUpperBound.branch row11677_large_left_left_covers row11677_large_left_right_covers

def row11677_large_right_left : List Nat := row11677_block_009 ++ row11677_block_010
theorem row11677_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11677 row11677_large_right_left :=
  Budget.RowUpperBound.branch row11677_block_009_covers row11677_block_010_covers

def row11677_large_right_right_right : List Nat := row11677_block_012 ++ row11677_block_013
theorem row11677_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11677 row11677_large_right_right_right :=
  Budget.RowUpperBound.branch row11677_block_012_covers row11677_block_013_covers

def row11677_large_right_right : List Nat := row11677_block_011 ++ row11677_large_right_right_right
theorem row11677_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11677 row11677_large_right_right :=
  Budget.RowUpperBound.branch row11677_block_011_covers row11677_large_right_right_right_covers

def row11677_large_right : List Nat := row11677_large_right_left ++ row11677_large_right_right
theorem row11677_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11677 row11677_large_right :=
  Budget.RowUpperBound.branch row11677_large_right_left_covers row11677_large_right_right_covers

def row11677_large : List Nat := row11677_large_left ++ row11677_large_right
theorem row11677_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11677 row11677_large :=
  Budget.RowUpperBound.branch row11677_large_left_covers row11677_large_right_covers

def row11677 : List Nat := row11677_large
theorem row11677_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11677 row11677 := row11677_large_covers

def row11681_small : List Nat := []
theorem row11681_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 11681 row11681_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row11681_block_005 : List Nat := []
theorem row11681_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 11681 row11681_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row11681_block_006 : List Nat := [11897]
def plan11681_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, (.secondFactor 19), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), (.secondFactor 7), .second, (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 13), .first, .possible, (.firstFactor 7), (.firstFactor 29), .first, (.secondFactor 7), .first, (.firstFactor 17), .first, .first, .first, (.secondFactor 7), .first, (.secondFactor 13), (.firstFactor 7), .first, (.firstFactor 11), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 17), .first, .first, .first, (.firstFactor 7), .first, .first, .second, (.firstFactor 7), .first, (.secondFactor 17), .first, (.secondFactor 7), (.secondFactor 19), .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, .first, .first, .first, (.firstFactor 13)]
set_option maxRecDepth 2048 in
theorem row11681_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 11681 row11681_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan11681_block_006
  · decide
  · rfl

def row11681_block_007 : List Nat := [12899, 13241, 13553, 13613, 14159]
def plan11681_block_007 : List Budget.Choice := [.first, .first, .second, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), (.secondFactor 19), .first, .first, .first, (.secondFactor 13), (.firstFactor 11), .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 19), .first, .first, .first, (.secondFactor 7), (.firstFactor 13), (.firstFactor 17), (.firstFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .possible, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, (.firstFactor 23), (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 19), .second, (.firstFactor 7), (.firstFactor 29), .first, (.secondFactor 13), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), (.firstFactor 19), .first, (.secondFactor 7), .possible, .first, (.firstFactor 23), .first, .first, .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .second, .first, (.firstFactor 17), .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 37), .first, .second, (.firstFactor 7), (.firstFactor 11), (.firstFactor 17), .first, .second, (.firstFactor 7), .first, (.secondFactor 7), .first, .possible, .first, (.secondFactor 17), .first, .first, .possible, (.secondFactor 13), .first, .first, (.firstFactor 29), .first, (.secondFactor 17), .first, .first, (.secondFactor 7), .first, (.firstFactor 31), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 31), (.secondFactor 17), .first, .first, .first, .first, .first, (.secondFactor 29), (.firstFactor 11), .first, (.firstFactor 7), .first, (.firstFactor 31), (.firstFactor 19), .first, .first, .first, .first, (.secondFactor 13), .first, .first, .second, (.firstFactor 29), .first, (.secondFactor 13), .first, .first, .second, (.firstFactor 13), (.firstFactor 11), .first, (.firstFactor 17), .first, (.firstFactor 7), .first, .first, .first, (.secondFactor 7), .possible, .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 19), .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row11681_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 11681 row11681_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan11681_block_007
  · decide
  · rfl

def row11681_block_008 : List Nat := []
theorem row11681_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 11681 row11681_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row11681_block_009 : List Nat := []
theorem row11681_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 11681 row11681_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row11681_block_010 : List Nat := []
theorem row11681_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 11681 row11681_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row11681_block_011 : List Nat := []
theorem row11681_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 11681 row11681_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row11681_block_012 : List Nat := []
theorem row11681_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 11681 row11681_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row11681_block_013 : List Nat := []
theorem row11681_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 11681 row11681_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row11681_large_left_left : List Nat := row11681_small ++ row11681_block_005
theorem row11681_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 11681 row11681_large_left_left :=
  Budget.RowUpperBound.branch row11681_small_covers row11681_block_005_covers

def row11681_large_left_right_right : List Nat := row11681_block_007 ++ row11681_block_008
theorem row11681_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 11681 row11681_large_left_right_right :=
  Budget.RowUpperBound.branch row11681_block_007_covers row11681_block_008_covers

def row11681_large_left_right : List Nat := row11681_block_006 ++ row11681_large_left_right_right
theorem row11681_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 11681 row11681_large_left_right :=
  Budget.RowUpperBound.branch row11681_block_006_covers row11681_large_left_right_right_covers

def row11681_large_left : List Nat := row11681_large_left_left ++ row11681_large_left_right
theorem row11681_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 11681 row11681_large_left :=
  Budget.RowUpperBound.branch row11681_large_left_left_covers row11681_large_left_right_covers

def row11681_large_right_left : List Nat := row11681_block_009 ++ row11681_block_010
theorem row11681_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 11681 row11681_large_right_left :=
  Budget.RowUpperBound.branch row11681_block_009_covers row11681_block_010_covers

def row11681_large_right_right_right : List Nat := row11681_block_012 ++ row11681_block_013
theorem row11681_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 11681 row11681_large_right_right_right :=
  Budget.RowUpperBound.branch row11681_block_012_covers row11681_block_013_covers

def row11681_large_right_right : List Nat := row11681_block_011 ++ row11681_large_right_right_right
theorem row11681_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 11681 row11681_large_right_right :=
  Budget.RowUpperBound.branch row11681_block_011_covers row11681_large_right_right_right_covers

def row11681_large_right : List Nat := row11681_large_right_left ++ row11681_large_right_right
theorem row11681_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 11681 row11681_large_right :=
  Budget.RowUpperBound.branch row11681_large_right_left_covers row11681_large_right_right_covers

def row11681_large : List Nat := row11681_large_left ++ row11681_large_right
theorem row11681_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 11681 row11681_large :=
  Budget.RowUpperBound.branch row11681_large_left_covers row11681_large_right_covers

def row11681 : List Nat := row11681_large
theorem row11681_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 11681 row11681 := row11681_large_covers

end ProjectEulerSolutions.PrimePairLarge
