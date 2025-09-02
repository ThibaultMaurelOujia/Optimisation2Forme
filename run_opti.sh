#!/usr/bin/env bash
set -euo pipefail
export LC_ALL=C
export LANG=C

STEP_M=0.0005  
STEP_P=0.002   
P_MIN=1e-3   
P_MAX=0.999
M_MIN=0.0     
M_MAX=0.12


die() { echo "ERROR: $*" >&2; exit 1; }

number_or_zero() {
  awk 'BEGIN{v="'"$1"'"; if(v+0==v) print v; else print 0.0}'
}
 
read_geo_val() { # $1=clé $2=fichier.geo
  local key="$1" geo="$2"
  awk -v k="$key" '
    BEGIN{FS="="}
    $0 ~ "^[[:space:]]*"k"[[:space:]]*=" {
      val=$2
      sub(/[;#].*$/,"",val)   
      gsub(/[[:space:]]/,"",val)
      print val; exit
    }' "$geo"
}
 
parse_log_J_and_grads() { # $1=logfile
  local log="$1"
  awk '
    /dJ\/dm/ {
      line=$0
      sub(/.*dJ\/dm[ \t]*=[ \t]*/,"",line)
      val=line
      sub(/[ \t].*$/,"",val)
      gm=val
    }
    /dJ\/dp/ {
      line=$0
      sub(/.*dJ\/dp[ \t]*=[ \t]*/,"",line)
      val=line
      sub(/[ \t].*$/,"",val)
      gp=val
    }
    /J/ {
      line=$0
      if (line ~ /J[ \t]*courant[ \t]*=/) {
        sub(/.*J[ \t]*courant[ \t]*=[ \t]*/,"",line)
      } else if (line ~ /J[ \t]*=/) {
        sub(/.*J[ \t]*=[ \t]*/,"",line)
      } else if (line ~ /J=/) {
        sub(/.*J=/,"",line)
      } else {
        line=""
      }
      if (line!="") {
        val=line
        sub(/[ \t].*$/,"",val)
        J=val
      }
    }
    {
      for (i=1; i<=NF; ++i) {
        if ($i ~ /^J=/) { v=$i; sub(/^J=/,"",v); J=v }
        if ($i ~ /^m=/) { v=$i; sub(/^m=/,"",v); gm=v }
        if ($i ~ /^p=/) { v=$i; sub(/^p=/,"",v); gp=v }
      }
    }
    END {
      if (!J)  J=0
      if (!gm) gm=0
      if (!gp) gp=0
      printf("%.16g %.16g %.16g\n", J+0, gm+0, gp+0)
    }
  ' "$log"
}
 
clamp() { # $1=val $2=lo $3=hi
  awk -v v="$1" -v lo="$2" -v hi="$3" 'BEGIN{ if(v<lo)v=lo; if(v>hi)v=hi; print v }'
}
 
replace_geo_numeric() { # $1=clé $2=nouvelle_val $3=fichier
  local key="$1" val="$2" file="$3" 
  sed -E -i '' -e "s/^([[:space:]]*${key}[[:space:]]*=[[:space:]]*)[^;#[:space:]]+/\1${val}/" "$file"
}
 
update_geo_from_grads() { # $1=geo_in $2=geo_out $3=grad_m $4=grad_p
  local geo_in="$1" geo_out="$2" gm="$3" gp="$4"

  cp -f "$geo_in" "$geo_out" || die "cp $geo_in -> $geo_out"
 
  local m_old p_old
  m_old="$(read_geo_val m "$geo_in")"; : "${m_old:=0.00}"
  p_old="$(read_geo_val p "$geo_in")"; : "${p_old:=0.40}"


  local s_m s_p
  s_m=$(awk -v g="$gm" 'BEGIN{print (g<0)?1:-1}')
  s_p=$(awk -v g="$gp" 'BEGIN{print (g<0)?1:-1}')

  local m_new p_new
  m_new=$(awk -v a="$m_old" -v s="$s_m" -v h="$STEP_M" 'BEGIN{printf("%.8f", a + s*h)}')
  p_new=$(awk -v a="$p_old" -v s="$s_p" -v h="$STEP_P" 'BEGIN{printf("%.8f", a + s*h)}')


  m_new="$(clamp "$m_new" "$M_MIN" "$M_MAX")"
  p_new="$(clamp "$p_new" "$P_MIN" "$P_MAX")"

  printf -v m_new "%.8f" "$m_new"
  printf -v p_new "%.8f" "$p_new"
 
  replace_geo_numeric "m" "$m_new" "$geo_out"
  replace_geo_numeric "p" "$p_new" "$geo_out"

  echo "[info] geo: m_old=$m_old -> m_new=$m_new ; p_old=$p_old -> p_new=$p_new"
}
 
update_params_txt() { # $1=txt_in $2=txt_out $3=m_new $4=p_new
  local txt_in="$1" txt_out="$2" m_new="$3" p_new="$4"

  cp -f "$txt_in" "$txt_out" || die "cp $txt_in -> $txt_out"
 
  sed -E -i '' -e "s/^([[:space:]]*m[[:space:]]*\*=?[[:space:]]*)[^#[:space:]]+/\1${m_new}/" "$txt_out"
  sed -E -i '' -e "s/^([[:space:]]*p[[:space:]]*\*=?[[:space:]]*)[^#[:space:]]+/\1${p_new}/" "$txt_out"
}


update_config_txt() { 
  local cfg="$1"

  awk -v OFS="" '
    function rindex(s, t,    i, pos) {
      pos = 0
      for (i = 1; i <= length(s); i++) if (substr(s, i, length(t)) == t) pos = i
      return pos
    }

    function rtrim(s) {
      sub(/[[:space:]]+$/, "", s)
      return s
    }

    function bump(s,    ss,slashPos,dotPos,i,ch,base,ext,u,numStr,tmp,isNum,num) {
      ss = rtrim(s)

      slashPos = 0; dotPos = 0
      for (i = 1; i <= length(ss); i++) {
        ch = substr(ss, i, 1)
        if (ch == "/") slashPos = i
        if (ch == ".") dotPos = i
      }

      if (dotPos > slashPos) {   
        base = substr(ss, 1, dotPos - 1)
        ext  = substr(ss, dotPos)
      } else {
        base = ss
        ext  = ""
      }
 
      u = rindex(base, "_")
      if (u == 0) return s

      numStr = substr(base, u + 1)
      tmp = numStr
      gsub(/[0-9]/, "", tmp)  
      isNum = (length(numStr) > 0 && length(tmp) == 0)

      if (!isNum) return s

      num = numStr + 0
      num++
      return substr(base, 1, u) "" num "" ext
    }

    function rewrite_line(key) {
      line = $0
      sub("^[[:space:]]*" key "[[:space:]]*=[[:space:]]*", "", line)
      newv = bump(line)
      print key " = " , newv
    }

    /^[[:space:]]*state_load_path[[:space:]]*=/  { rewrite_line("state_load_path");  next }
    /^[[:space:]]*state_save_path[[:space:]]*=/  { rewrite_line("state_save_path");  next }
    /^[[:space:]]*shape_param_file[[:space:]]*=/ { rewrite_line("shape_param_file"); next }
    /^[[:space:]]*shape_log_file[[:space:]]*=/   { rewrite_line("shape_log_file");   next }
    /^[[:space:]]*mesh_name[[:space:]]*=/        { rewrite_line("mesh_name");        next }

    { print }
  ' "$cfg" > "$cfg.tmp" && mv "$cfg.tmp" "$cfg"
}




run_solver() {
  ./navier_stokes
}


run_gmsh() { # $1=geo_out $2=msh_out
  local geo="$1" msh="$2"
  gmsh "$geo" -2 -format msh2 -o "$msh"
}


one_step() { # $1 = i
  local i="$1"
  local j=$((i+1))

  local LOG_IN="mesh_opti/shape_opt_${i}.log"
  local GEO_IN="mesh_opti/naca_2D_shape_${i}.geo"
  local GEO_OUT="mesh_opti/naca_2D_shape_${j}.geo"
  local MSH_OUT="mesh_opti/naca_2D_shape_${j}.msh"
  local TXT_IN="mesh_opti/naca_2D_shape_params_${i}.txt"
  local TXT_OUT="mesh_opti/naca_2D_shape_params_${j}.txt"
  local CFG="config.txt"


  run_solver


  [[ -f "$LOG_IN" ]] || die "Log introuvable: $LOG_IN"
  read -r J gm gp < <(parse_log_J_and_grads "$LOG_IN")
  gm="$(number_or_zero "$gm")"
  gp="$(number_or_zero "$gp")"
  echo "[info] lu depuis log: J=$J dJ/dm=$gm dJ/dp=$gp"


  [[ -f "$GEO_IN" ]] || die "Geo introuvable: $GEO_IN"
  update_geo_from_grads "$GEO_IN" "$GEO_OUT" "$gm" "$gp"


  local m_new p_new
  m_new="$(read_geo_val m "$GEO_OUT")"
  p_new="$(read_geo_val p "$GEO_OUT")"
  printf -v m_new "%.8f" "$m_new"
  printf -v p_new "%.8f" "$p_new"


  run_gmsh "$GEO_OUT" "$MSH_OUT"


  [[ -f "$TXT_IN" ]] || die "Params introuvable: $TXT_IN"
  update_params_txt "$TXT_IN" "$TXT_OUT" "$m_new" "$p_new"


  [[ -f "$CFG" ]] || die "config.txt introuvable"
  update_config_txt "$CFG" "$i" "$j"
}


main() {
  local N="${1:-1}"   
  local start="${2:-0}" 

  for ((k=start; k<start+N; ++k)); do
    echo "=== itération $k -> $((k+1)) ==="
    one_step "$k"
  done
}

main "$@"
