import ProjectEulerSolutions.Lib.PrimePairLarge.Bounds

namespace ProjectEulerSolutions.PrimePairLarge
open PrimePairCertificate

def row12043_small : List Nat := []
theorem row12043_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12043 row12043_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12043_block_005 : List Nat := []
theorem row12043_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12043 row12043_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12043_block_006 : List Nat := []
def plan12043_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, (.secondFactor 17), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 31), .first, .first, (.firstFactor 11), .first, (.secondFactor 17), .first, .first, (.firstFactor 7), .first, (.secondFactor 37), .first, .first, .first, (.firstFactor 13), .first, (.firstFactor 11), .first, .first, (.secondFactor 13), .first, (.secondFactor 7), .first]
set_option maxRecDepth 2048 in
theorem row12043_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12043 row12043_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan12043_block_006
  · decide
  · rfl

def row12043_block_007 : List Nat := []
def plan12043_block_007 : List Budget.Choice := [(.firstFactor 13), .first, .first, (.secondFactor 7), (.secondFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 19), .second, .first, .first, (.secondFactor 19), (.secondFactor 13), .second, .first, (.firstFactor 13), (.firstFactor 31), (.firstFactor 17), .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 11), .first, .first, (.firstFactor 7), .first, (.secondFactor 23), (.secondFactor 17), (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.firstFactor 29), (.firstFactor 19), (.firstFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.secondFactor 13), (.firstFactor 11), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 13), (.secondFactor 37), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 7), (.firstFactor 11), (.firstFactor 19), .first, (.firstFactor 7), .first, .first, (.secondFactor 19), .first, .second, .first, (.secondFactor 37), (.firstFactor 7), (.secondFactor 7), .first, (.secondFactor 31), .first, .first, .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), .first, (.secondFactor 19), .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 13), .first, (.secondFactor 17), (.secondFactor 29), .first, (.firstFactor 11), .first, .first, (.firstFactor 17), .first, (.secondFactor 23), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 19), .first, .first, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12043_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12043 row12043_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12043_block_007
  · decide
  · rfl

def row12043_block_008 : List Nat := []
theorem row12043_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12043 row12043_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12043_block_009 : List Nat := []
theorem row12043_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12043 row12043_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12043_block_010 : List Nat := []
theorem row12043_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12043 row12043_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12043_block_011 : List Nat := []
theorem row12043_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12043 row12043_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12043_block_012 : List Nat := []
theorem row12043_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12043 row12043_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12043_block_013 : List Nat := []
theorem row12043_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12043 row12043_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12043_large_left_left : List Nat := row12043_small ++ row12043_block_005
theorem row12043_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12043 row12043_large_left_left :=
  Budget.RowUpperBound.branch row12043_small_covers row12043_block_005_covers

def row12043_large_left_right_right : List Nat := row12043_block_007 ++ row12043_block_008
theorem row12043_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12043 row12043_large_left_right_right :=
  Budget.RowUpperBound.branch row12043_block_007_covers row12043_block_008_covers

def row12043_large_left_right : List Nat := row12043_block_006 ++ row12043_large_left_right_right
theorem row12043_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12043 row12043_large_left_right :=
  Budget.RowUpperBound.branch row12043_block_006_covers row12043_large_left_right_right_covers

def row12043_large_left : List Nat := row12043_large_left_left ++ row12043_large_left_right
theorem row12043_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12043 row12043_large_left :=
  Budget.RowUpperBound.branch row12043_large_left_left_covers row12043_large_left_right_covers

def row12043_large_right_left : List Nat := row12043_block_009 ++ row12043_block_010
theorem row12043_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12043 row12043_large_right_left :=
  Budget.RowUpperBound.branch row12043_block_009_covers row12043_block_010_covers

def row12043_large_right_right_right : List Nat := row12043_block_012 ++ row12043_block_013
theorem row12043_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12043 row12043_large_right_right_right :=
  Budget.RowUpperBound.branch row12043_block_012_covers row12043_block_013_covers

def row12043_large_right_right : List Nat := row12043_block_011 ++ row12043_large_right_right_right
theorem row12043_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12043 row12043_large_right_right :=
  Budget.RowUpperBound.branch row12043_block_011_covers row12043_large_right_right_right_covers

def row12043_large_right : List Nat := row12043_large_right_left ++ row12043_large_right_right
theorem row12043_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12043 row12043_large_right :=
  Budget.RowUpperBound.branch row12043_large_right_left_covers row12043_large_right_right_covers

def row12043_large : List Nat := row12043_large_left ++ row12043_large_right
theorem row12043_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12043 row12043_large :=
  Budget.RowUpperBound.branch row12043_large_left_covers row12043_large_right_covers

def row12043 : List Nat := row12043_large
theorem row12043_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12043 row12043 := row12043_large_covers

def row12049_small : List Nat := []
theorem row12049_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12049 row12049_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12049_block_005 : List Nat := []
theorem row12049_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12049 row12049_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12049_block_006 : List Nat := []
def plan12049_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, (.firstFactor 29), .first, .first, (.firstFactor 23), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 13), .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .second, .first, (.firstFactor 7), .first, (.firstFactor 11), (.secondFactor 7), .first, (.secondFactor 13), .first]
set_option maxRecDepth 2048 in
theorem row12049_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12049 row12049_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan12049_block_006
  · decide
  · rfl

def row12049_block_007 : List Nat := [12739]
def plan12049_block_007 : List Budget.Choice := [.second, (.firstFactor 31), .first, (.firstFactor 17), (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, (.firstFactor 13), .first, (.firstFactor 11), (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.firstFactor 29), .first, .first, .first, .first, .first, .first, .first, .second, .first, .first, (.firstFactor 19), .possible, .first, (.firstFactor 17), .second, .second, .first, (.firstFactor 23), .first, .first, .first, .first, (.firstFactor 11), (.secondFactor 7), .first, .first, .first, (.firstFactor 11), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 29), (.firstFactor 11), (.secondFactor 7), .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 13), .first, .first, .first, (.firstFactor 17), .second, (.firstFactor 13), .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, .first, .first, .first, .first, .first, .first, .first, (.firstFactor 17), (.firstFactor 19), (.firstFactor 7), .first, .first, .first, .first, .first, (.firstFactor 7), .first, (.firstFactor 7), .first, (.firstFactor 37), (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, .first, .second, .first, .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), (.secondFactor 31), .first, .first, (.firstFactor 23), (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, (.firstFactor 7), .first, .first, .second, .first, .first, .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 13), .first, .first, (.firstFactor 23), .first, .first, (.firstFactor 31), .first, .second, (.firstFactor 11), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12049_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12049 row12049_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12049_block_007
  · decide
  · rfl

def row12049_block_008 : List Nat := []
theorem row12049_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12049 row12049_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12049_block_009 : List Nat := []
theorem row12049_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12049 row12049_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12049_block_010 : List Nat := []
theorem row12049_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12049 row12049_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12049_block_011 : List Nat := []
theorem row12049_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12049 row12049_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12049_block_012 : List Nat := []
theorem row12049_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12049 row12049_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12049_block_013 : List Nat := []
theorem row12049_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12049 row12049_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12049_large_left_left : List Nat := row12049_small ++ row12049_block_005
theorem row12049_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12049 row12049_large_left_left :=
  Budget.RowUpperBound.branch row12049_small_covers row12049_block_005_covers

def row12049_large_left_right_right : List Nat := row12049_block_007 ++ row12049_block_008
theorem row12049_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12049 row12049_large_left_right_right :=
  Budget.RowUpperBound.branch row12049_block_007_covers row12049_block_008_covers

def row12049_large_left_right : List Nat := row12049_block_006 ++ row12049_large_left_right_right
theorem row12049_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12049 row12049_large_left_right :=
  Budget.RowUpperBound.branch row12049_block_006_covers row12049_large_left_right_right_covers

def row12049_large_left : List Nat := row12049_large_left_left ++ row12049_large_left_right
theorem row12049_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12049 row12049_large_left :=
  Budget.RowUpperBound.branch row12049_large_left_left_covers row12049_large_left_right_covers

def row12049_large_right_left : List Nat := row12049_block_009 ++ row12049_block_010
theorem row12049_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12049 row12049_large_right_left :=
  Budget.RowUpperBound.branch row12049_block_009_covers row12049_block_010_covers

def row12049_large_right_right_right : List Nat := row12049_block_012 ++ row12049_block_013
theorem row12049_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12049 row12049_large_right_right_right :=
  Budget.RowUpperBound.branch row12049_block_012_covers row12049_block_013_covers

def row12049_large_right_right : List Nat := row12049_block_011 ++ row12049_large_right_right_right
theorem row12049_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12049 row12049_large_right_right :=
  Budget.RowUpperBound.branch row12049_block_011_covers row12049_large_right_right_right_covers

def row12049_large_right : List Nat := row12049_large_right_left ++ row12049_large_right_right
theorem row12049_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12049 row12049_large_right :=
  Budget.RowUpperBound.branch row12049_large_right_left_covers row12049_large_right_right_covers

def row12049_large : List Nat := row12049_large_left ++ row12049_large_right
theorem row12049_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12049 row12049_large :=
  Budget.RowUpperBound.branch row12049_large_left_covers row12049_large_right_covers

def row12049 : List Nat := row12049_large
theorem row12049_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12049 row12049 := row12049_large_covers

def row12071_small : List Nat := []
theorem row12071_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12071 row12071_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12071_block_005 : List Nat := []
theorem row12071_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12071 row12071_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12071_block_006 : List Nat := [12113, 12377]
def plan12071_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .first, .first, (.firstFactor 13), .first, .first, .possible, .first, (.secondFactor 7), (.firstFactor 17), .first, (.firstFactor 19), .first, (.firstFactor 23), (.firstFactor 11), .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, (.firstFactor 7), (.firstFactor 11), .first, (.secondFactor 23), .first, .first, .first, (.secondFactor 19), .first, (.firstFactor 7), .first, .possible, .first, .first, (.firstFactor 11), .first, (.firstFactor 13)]
set_option maxRecDepth 2048 in
theorem row12071_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12071 row12071_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan12071_block_006
  · decide
  · rfl

def row12071_block_007 : List Nat := [12497, 12953, 13421, 13499, 13709, 13763, 13877]
def plan12071_block_007 : List Budget.Choice := [.first, .first, (.secondFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 7), .first, (.firstFactor 13), .possible, (.firstFactor 19), .first, .first, (.secondFactor 13), .first, .first, .first, .first, (.firstFactor 13), .first, .first, .first, .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 7), (.firstFactor 13), .first, (.firstFactor 17), (.secondFactor 19), (.firstFactor 31), .first, .first, (.secondFactor 17), .first, .first, (.secondFactor 31), .first, .first, .first, (.firstFactor 29), .first, (.firstFactor 7), .second, .first, .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 7), .first, .first, (.secondFactor 13), .first, .second, (.secondFactor 7), .possible, (.firstFactor 13), .first, .first, .first, (.secondFactor 7), .first, .first, .second, .first, .first, (.firstFactor 13), .first, (.firstFactor 37), .first, .first, .first, (.firstFactor 7), (.secondFactor 7), (.secondFactor 17), (.firstFactor 11), .first, (.secondFactor 7), .first, (.firstFactor 23), .first, .first, .first, (.firstFactor 7), .first, .first, (.firstFactor 7), (.secondFactor 19), .first, (.firstFactor 11), .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, .first, (.firstFactor 7), .first, .first, .first, .possible, .first, .second, (.firstFactor 11), (.secondFactor 13), (.secondFactor 19), .first, (.firstFactor 29), .possible, .first, (.firstFactor 7), .first, (.secondFactor 29), .first, (.firstFactor 17), .first, .first, (.secondFactor 7), (.firstFactor 31), .first, .first, (.firstFactor 7), .first, (.firstFactor 17), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .possible, .first, (.firstFactor 11), .first, .first, .first, (.firstFactor 19), .first, .possible, (.firstFactor 17), .first, (.secondFactor 23), .first, .second, .first, .first, (.firstFactor 7), .first, .possible, .first, (.firstFactor 17), (.firstFactor 7), .first, (.secondFactor 7), .second, .first, (.secondFactor 13), .first, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12071_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12071 row12071_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12071_block_007
  · decide
  · rfl

def row12071_block_008 : List Nat := []
theorem row12071_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12071 row12071_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12071_block_009 : List Nat := []
theorem row12071_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12071 row12071_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12071_block_010 : List Nat := []
theorem row12071_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12071 row12071_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12071_block_011 : List Nat := []
theorem row12071_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12071 row12071_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12071_block_012 : List Nat := []
theorem row12071_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12071 row12071_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12071_block_013 : List Nat := []
theorem row12071_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12071 row12071_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12071_large_left_left : List Nat := row12071_small ++ row12071_block_005
theorem row12071_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12071 row12071_large_left_left :=
  Budget.RowUpperBound.branch row12071_small_covers row12071_block_005_covers

def row12071_large_left_right_right : List Nat := row12071_block_007 ++ row12071_block_008
theorem row12071_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12071 row12071_large_left_right_right :=
  Budget.RowUpperBound.branch row12071_block_007_covers row12071_block_008_covers

def row12071_large_left_right : List Nat := row12071_block_006 ++ row12071_large_left_right_right
theorem row12071_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12071 row12071_large_left_right :=
  Budget.RowUpperBound.branch row12071_block_006_covers row12071_large_left_right_right_covers

def row12071_large_left : List Nat := row12071_large_left_left ++ row12071_large_left_right
theorem row12071_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12071 row12071_large_left :=
  Budget.RowUpperBound.branch row12071_large_left_left_covers row12071_large_left_right_covers

def row12071_large_right_left : List Nat := row12071_block_009 ++ row12071_block_010
theorem row12071_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12071 row12071_large_right_left :=
  Budget.RowUpperBound.branch row12071_block_009_covers row12071_block_010_covers

def row12071_large_right_right_right : List Nat := row12071_block_012 ++ row12071_block_013
theorem row12071_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12071 row12071_large_right_right_right :=
  Budget.RowUpperBound.branch row12071_block_012_covers row12071_block_013_covers

def row12071_large_right_right : List Nat := row12071_block_011 ++ row12071_large_right_right_right
theorem row12071_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12071 row12071_large_right_right :=
  Budget.RowUpperBound.branch row12071_block_011_covers row12071_large_right_right_right_covers

def row12071_large_right : List Nat := row12071_large_right_left ++ row12071_large_right_right
theorem row12071_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12071 row12071_large_right :=
  Budget.RowUpperBound.branch row12071_large_right_left_covers row12071_large_right_right_covers

def row12071_large : List Nat := row12071_large_left ++ row12071_large_right
theorem row12071_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12071 row12071_large :=
  Budget.RowUpperBound.branch row12071_large_left_covers row12071_large_right_covers

def row12071 : List Nat := row12071_large
theorem row12071_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12071 row12071 := row12071_large_covers

def row12073_small : List Nat := []
theorem row12073_small_covers : Budget.RowUpperBound PrimePairVertices.tree_small 26034 12073 row12073_small :=
  Budget.empty_row_before _ _ _ 8389 (fun q member => (Bounds.small q member).2) (by decide)

def row12073_block_005 : List Nat := []
theorem row12073_block_005_covers : Budget.RowUpperBound PrimePairVertices.tree_block_005 26034 12073 row12073_block_005 :=
  Budget.empty_row_before _ _ _ 10370 (fun q member => (Bounds.block_005 q member).2) (by decide)

def row12073_block_006 : List Nat := [12373]
def plan12073_block_006 : List Budget.Choice := [.outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .second, .first, .first, (.firstFactor 37), .first, .first, .first, .first, .first, .first, (.secondFactor 17), .first, .first, (.firstFactor 7), .first, .first, .first, .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, .second, (.firstFactor 13), .first, .first, (.firstFactor 23), .first, .possible, .first, (.firstFactor 7), (.firstFactor 31), .first, (.firstFactor 17), .first]
set_option maxRecDepth 2048 in
theorem row12073_block_006_covers : Budget.RowUpperBound PrimePairVertices.tree_block_006 26034 12073 row12073_block_006 := by
  apply Budget.row_upper_of_plan _ _ _ plan12073_block_006
  · decide
  · rfl

def row12073_block_007 : List Nat := [12451, 12703, 13147]
def plan12073_block_007 : List Budget.Choice := [(.firstFactor 7), (.secondFactor 37), .first, .possible, (.firstFactor 13), .first, .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7), (.firstFactor 37), .first, (.firstFactor 31), .first, (.firstFactor 7), (.firstFactor 11), .first, (.firstFactor 13), (.firstFactor 23), (.secondFactor 13), .first, .first, .first, .first, .first, .first, (.secondFactor 7), .possible, .first, (.secondFactor 19), (.secondFactor 7), .first, (.firstFactor 7), (.firstFactor 19), (.secondFactor 7), .first, (.firstFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 7), .second, .second, .first, .first, (.secondFactor 7), .first, .first, (.firstFactor 17), .first, .first, .first, .first, (.firstFactor 7), .first, (.secondFactor 17), .first, .first, (.firstFactor 13), .first, (.firstFactor 7), (.firstFactor 23), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), (.secondFactor 37), .first, .first, .first, .first, .possible, .first, (.firstFactor 13), .first, (.firstFactor 23), (.firstFactor 7), (.secondFactor 13), .first, .first, (.firstFactor 7), .first, .first, .first, .first, .first, (.secondFactor 19), (.firstFactor 29), (.firstFactor 23), .first, (.firstFactor 11), .first, .first, (.secondFactor 13), .first, .first, .first, .second, (.secondFactor 7), (.secondFactor 13), .first, (.firstFactor 37), .first, .first, .first, .first, (.secondFactor 31), .first, .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, .first, (.firstFactor 11), (.firstFactor 7), .first, .first, (.firstFactor 13), (.firstFactor 17), .first, .first, .first, (.firstFactor 7), .second, .first, (.firstFactor 31), .first, .first, .first, .first, (.firstFactor 7), (.secondFactor 13), .first, .first, .first, .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .first, (.secondFactor 7), .first, .first, (.secondFactor 7), .first, (.firstFactor 31), .first, .first, (.firstFactor 19), .first, .first, (.firstFactor 11), .first, (.firstFactor 7), .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside, .outside]
set_option maxRecDepth 2048 in
theorem row12073_block_007_covers : Budget.RowUpperBound PrimePairVertices.tree_block_007 26034 12073 row12073_block_007 := by
  apply Budget.row_upper_of_plan _ _ _ plan12073_block_007
  · decide
  · rfl

def row12073_block_008 : List Nat := []
theorem row12073_block_008_covers : Budget.RowUpperBound PrimePairVertices.tree_block_008 26034 12073 row12073_block_008 :=
  Budget.empty_row_after _ _ _ 14466 (fun q member => (Bounds.block_008 q member).1) (by decide)

def row12073_block_009 : List Nat := []
theorem row12073_block_009_covers : Budget.RowUpperBound PrimePairVertices.tree_block_009 26034 12073 row12073_block_009 :=
  Budget.empty_row_after _ _ _ 16514 (fun q member => (Bounds.block_009 q member).1) (by decide)

def row12073_block_010 : List Nat := []
theorem row12073_block_010_covers : Budget.RowUpperBound PrimePairVertices.tree_block_010 26034 12073 row12073_block_010 :=
  Budget.empty_row_after _ _ _ 18562 (fun q member => (Bounds.block_010 q member).1) (by decide)

def row12073_block_011 : List Nat := []
theorem row12073_block_011_covers : Budget.RowUpperBound PrimePairVertices.tree_block_011 26034 12073 row12073_block_011 :=
  Budget.empty_row_after _ _ _ 20610 (fun q member => (Bounds.block_011 q member).1) (by decide)

def row12073_block_012 : List Nat := []
theorem row12073_block_012_covers : Budget.RowUpperBound PrimePairVertices.tree_block_012 26034 12073 row12073_block_012 :=
  Budget.empty_row_after _ _ _ 22658 (fun q member => (Bounds.block_012 q member).1) (by decide)

def row12073_block_013 : List Nat := []
theorem row12073_block_013_covers : Budget.RowUpperBound PrimePairVertices.tree_block_013 26034 12073 row12073_block_013 :=
  Budget.empty_row_after _ _ _ 24706 (fun q member => (Bounds.block_013 q member).1) (by decide)

def row12073_large_left_left : List Nat := row12073_small ++ row12073_block_005
theorem row12073_large_left_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_left 26034 12073 row12073_large_left_left :=
  Budget.RowUpperBound.branch row12073_small_covers row12073_block_005_covers

def row12073_large_left_right_right : List Nat := row12073_block_007 ++ row12073_block_008
theorem row12073_large_left_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right_right 26034 12073 row12073_large_left_right_right :=
  Budget.RowUpperBound.branch row12073_block_007_covers row12073_block_008_covers

def row12073_large_left_right : List Nat := row12073_block_006 ++ row12073_large_left_right_right
theorem row12073_large_left_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left_right 26034 12073 row12073_large_left_right :=
  Budget.RowUpperBound.branch row12073_block_006_covers row12073_large_left_right_right_covers

def row12073_large_left : List Nat := row12073_large_left_left ++ row12073_large_left_right
theorem row12073_large_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_left 26034 12073 row12073_large_left :=
  Budget.RowUpperBound.branch row12073_large_left_left_covers row12073_large_left_right_covers

def row12073_large_right_left : List Nat := row12073_block_009 ++ row12073_block_010
theorem row12073_large_right_left_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_left 26034 12073 row12073_large_right_left :=
  Budget.RowUpperBound.branch row12073_block_009_covers row12073_block_010_covers

def row12073_large_right_right_right : List Nat := row12073_block_012 ++ row12073_block_013
theorem row12073_large_right_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right_right 26034 12073 row12073_large_right_right_right :=
  Budget.RowUpperBound.branch row12073_block_012_covers row12073_block_013_covers

def row12073_large_right_right : List Nat := row12073_block_011 ++ row12073_large_right_right_right
theorem row12073_large_right_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right_right 26034 12073 row12073_large_right_right :=
  Budget.RowUpperBound.branch row12073_block_011_covers row12073_large_right_right_right_covers

def row12073_large_right : List Nat := row12073_large_right_left ++ row12073_large_right_right
theorem row12073_large_right_covers : Budget.RowUpperBound PrimePairVertices.tree_large_right 26034 12073 row12073_large_right :=
  Budget.RowUpperBound.branch row12073_large_right_left_covers row12073_large_right_right_covers

def row12073_large : List Nat := row12073_large_left ++ row12073_large_right
theorem row12073_large_covers : Budget.RowUpperBound PrimePairVertices.tree_large 26034 12073 row12073_large :=
  Budget.RowUpperBound.branch row12073_large_left_covers row12073_large_right_covers

def row12073 : List Nat := row12073_large
theorem row12073_covers : Budget.RowUpperBound PrimePairVertices.largeTree 26034 12073 row12073 := row12073_large_covers

end ProjectEulerSolutions.PrimePairLarge
