extern crate imgui;
extern crate glfw;
extern crate notify;

use notify::{Watcher, RecursiveMode, RawEvent, raw_watcher, RecommendedWatcher};
use glfw::{Glfw, Action, Context, Key, MouseButton};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::mem::{size_of, size_of_val};
use std::time::Instant;
use std::ffi::CString;
use gl33::*;
use imgui::Context as ImContext;

/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////

type Vertex = [f32; 2];

const VERTICES: [Vertex; 6] =
    [[-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]]; // a quad

const VERT_SHADER: &str = r#"#version 330 core
    layout (location = 0) in vec2 pos;
    void main()
    {
        gl_Position = vec4(pos, 0.0, 1.0);
    }
"#;

const FRAG_SHADER: &str = r#"#version 330 core
    uniform vec2 size;
    uniform float time;
    uniform vec4 mouse;

    out vec4 final_color;

    // Created by Stephane Cuillerdier - @Aiekick/2019
    // License Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported License.

    // https://www.shadertoy.com/view/WslXWr

    // mouse x for cam rotation
    // mouse y for change count slices

    #define ANGLE_CUTTER
    //#define RADIUS_CUTTER

    // a bit noisy
    //#define HEIGHT_CUTTER

    float countSlices;

    float fullAtan(vec2 p)
    {
        return step(0.0,-p.x)*3.14159 + sign(p.x) * atan(p.x, sign(p.x) * p.y);
    }

    float julia(vec2 p, vec2 v)
    {
        vec2 z = p;
        vec2 c = v;
        float k = 1., h = 1.0;
        for (int i=0;i<5;i++)
        {
            h *= 4.*k;
            k = dot(z,z);
            if(k > 4.) break;
            z = vec2(z.x * z.x - z.y * z.y, 2. * z.x * z.y) + c;
        }
        return sqrt(k/h)*log(k);
    }

    float fractusSliced(vec3 p, float a, float n, float oa)
    {
    #ifdef HEIGHT_CUTTER
        p.y += oa/n;
        float cutterHeight = abs(abs(fract(p.y*n)-0.5) - 0.225)-0.225;
        p.y = floor(p.y*n)/n;
    #endif
    #ifdef ANGLE_CUTTER
        a += oa/n;
        float cutterAngle = abs(abs(fract(a*n)-0.5) - 0.225)-0.225;
        a = floor(a*n)/n;
    #endif
        vec2 c = vec2(mix(0.2, -0.5, sin(a * 2.)), mix(0.5, 0.0, sin(a * 3.)));
        float path = length(p.xz) - 3.;
    #ifdef RADIUS_CUTTER
        path += oa/n;
        float cutterRadius = abs(abs(fract(path*n)-0.5) - 0.225)-0.225;
        path = floor(path*n)/n;
    #endif
        vec2 rev = vec2(path, p.y);
        float aa = a + time;
        rev *= mat2(cos(aa),-sin(aa),sin(aa),cos(aa)); // rot near axis y
        float d = julia(rev, c);
    #ifdef HEIGHT_CUTTER
        d = max(cutterHeight,d);
    #endif
    #ifdef ANGLE_CUTTER
        d = max(cutterAngle,d);
    #endif
    #ifdef RADIUS_CUTTER
        d = max(cutterRadius,d);
    #endif
        return d;
    }

    float dfFractus(vec3 p)
    {
        float a = fullAtan(p.xz);
        return min(
            fractusSliced(p,a,countSlices,0.),
            fractusSliced(p,a,countSlices,0.5));
    }

    float map(vec3 p)
    {
        return min(
            p.y + 2.5 - sin(dot(p.xz,p.xz)*0.25 - time)*0.5,
            dfFractus(p));
    }

    float march( in vec3 ro, in vec3 rd )
    {
        float s = 1.;
        float d = 0.;
        for( int i=0; i<200; i++ )
        {
            if(
                //d*d/s>1e6||
                abs(s)<0.0025*(d*.125 + 1.)|| // shane formula
                d>100. ) break;
            s = map( ro+rd*d );
            d += s * 0.1;
        }
        return d;
    }

    vec3 getNor( vec3 p, float k )
    {
        vec3 eps = vec3( k, 0., 0. );
        vec3 nor = vec3(
            map(p+eps.xyy) - map(p-eps.xyy),
            map(p+eps.yxy) - map(p-eps.yxy),
            map(p+eps.yyx) - map(p-eps.yyx) );
        return normalize(nor);
    }

    // iq code
    float getSha( in vec3 ro, in vec3 rd, in float hn)
    {
        float res = 1.0;
        float t = 0.0005;
        float h = 1.0;
        for( int i=0; i<40; i++ )
        {
            h = map(ro + rd*t);
            res = min( res, hn*h/t );
            t += clamp( h, 0.02, 2.0 );
        }
        return clamp(res,0.0,1.0);
    }

    // iq code
    float getAO( in vec3 p, in vec3 nor )
    {
        float occ = 0.0;
        float sca = 1.0;
        for( int i=0; i<5; i++ )
        {
            float hr = 0.01 + 0.12 * float(i)/4.0;
            vec3 aopos =  nor * hr + p;
            float dd = map( aopos );
            occ += -(dd-hr)*sca;
            sca *= 0.95;
        }
        return clamp( 1.0 - 3.0*occ, 0.0, 1.0 );
    }

    // shane code
    // Tri-Planar blending function. Based on an old Nvidia tutorial.
    vec3 tex3D( sampler2D tex, in vec3 p, in vec3 n ){

        //return cellTileColor(p);

        n = max((abs(n) - 0.2)*7., 0.001); // n = max(abs(n), 0.001), etc.
        n /= (n.x + n.y + n.z );
        return (texture(tex, p.yz)*n.x + texture(tex, p.zx)*n.y + texture(tex, p.xy)*n.z).xyz;
    }

    // shane code
    // Texture bump mapping. Four tri-planar lookups, or 12 texture lookups in total. I tried to
    // make it as concise as possible. Whether that translates to speed, or not, I couldn't say.
    vec3 texBump( sampler2D tx, in vec3 p, in vec3 n, float bf){

        const vec2 e = vec2(0.002, 0);

        // Three gradient vectors rolled into a matrix, constructed with offset greyscale texture values.
        mat3 m = mat3( tex3D(tx, p - e.xyy, n), tex3D(tx, p - e.yxy, n), tex3D(tx, p - e.yyx, n));

        vec3 g = vec3(0.299, 0.587, 0.114)*m; // Converting to greyscale.
        g = (g - dot(tex3D(tx,  p , n), vec3(0.299, 0.587, 0.114)) )/e.x; g -= n*dot(n, g);

        return normalize( n + g*bf ); // Bumped normal. "bf" - bump factor.
    }

    vec3 light( in vec3 ld, in vec3 lc, in vec3 tex, in vec3 n, in vec3 rd )
    {
        float diff = pow(dot(n, -ld) * .5 + .5,2.0);
        float spe = pow(max(dot(-rd, reflect(ld, n)), 0.0), 32.0);

        return (tex * diff + spe) * lc * 1.5;
    }

    vec3 GetRainBow(float r)
    {
        int i = int(3.*fract(r));
        vec4 c = vec4(.25);
        c[(i+1)%3] += r = fract(3.*r);
        c[i] += 1.-r;
        return c.rgb;
    }

    vec3 shade( in vec3 ro , in vec3 rd, in float d )
    {
        vec3 p = ro + rd * d;

        vec3 n = getNor(p, 0.0001);

        vec3 d1 = -normalize(vec3(1,2.5,3));
        vec3 d2 = -normalize(vec3(5,3.5,0.5));

        vec3 texCol = GetRainBow(sin(fullAtan(p.xz)) * 0.5 + 0.5);

        float sha = 0.5 + 0.5 * getSha(p, n, 2.0);
        float ao = getAO(p, n);

        vec3 l1 = light(d1, vec3(1,1,1), texCol, n, rd);
        vec3 l2 = light(d2, vec3(1.2), texCol, n, rd);

        vec3 col = texCol * 0.2 * ao + (l1 + l2) * sha;

        return mix(col, vec3(0), 1.0-exp(-0.005*d*d));
    }

    void mainImage( out vec4 fragColor, in vec2 fragCoord )
    {
        countSlices = 30.;
        float ca = 3.6;

        if (mouse.x>0.0)
        {
            countSlices = floor(mouse.y / size.y * 29.) + 1.;
            ca = mouse.x / size.x * -6.28318;
        }

        vec2 uv = (fragCoord*2. - size)/min(size.x, size.y);
        vec3 ro = vec3(cos(ca), 8., sin(ca)); ro.xz *= 8.;
        vec3 rov = normalize(-ro);
        vec3 u = normalize(cross(vec3(0,1,0),rov));
        vec3 v = cross(rov,u);
        vec3 rd = normalize(rov + 0.5*uv.x*u + 0.5*uv.y*v);

        float d = march(ro, rd);

        vec3 col = shade(ro, rd, d);

        fragColor = vec4(sqrt(col*col*1.2),1.0);
    }

    void main()
    {
        final_color = vec4(0);
        mainImage(final_color, gl_FragCoord.xy);
        final_color.a = 1.0;
    }
"#;

/////////////////////////////////////////////////////
/////////////////////////////////////////////////////
/////////////////////////////////////////////////////

pub struct MainFrame
{
    m_glfw:Glfw,
    m_size_uniform_loc:i32,
    m_time_uniform_loc:i32,
    m_mouse_uniform_loc:i32
}

impl MainFrame
{
    pub fn new() -> Self
    {
        glfw::WindowHint::ContextVersion(3, 3);
        Self {
            m_glfw: glfw::init(glfw::FAIL_ON_ERRORS).unwrap(),
            m_size_uniform_loc: 0,
            m_time_uniform_loc: 0,
            m_mouse_uniform_loc: 0
        }
    }

    pub fn display(&mut self, title:&str, width:u32, height:u32)
    {
        let (mut window, events) =
            self.m_glfw.create_window(width, height, title, glfw::WindowMode::Windowed)
            .expect("Failed to create GLFW window.");

        window.make_current();
        window.set_key_polling(true);
        window.set_size_polling(true);
        window.set_mouse_button_polling(true);
        window.set_cursor_pos_polling(true);

        let gl = self.prepare();

        let mut imgui = ImContext::create();

        let mut uniform_size = (0f32, 0f32);
        let mut uniform_time = 0f32;

        let fbo_size = window.get_framebuffer_size();
        uniform_size.0 = fbo_size.0 as f32;
        uniform_size.1 = fbo_size.1 as f32;

        let mut mouse = (0f32,0f32,0f32,0f32);
        
        let mut tmp_mouse_pos = (0f64,0f64);
        let mut tmp_mouse_left_pressed = false;

        while !window.should_close()
        {
            let now = Instant::now();

            // Poll for and process events
            self.m_glfw.poll_events();
            for (_, event) in glfw::flush_messages(&events)
            {
                //println!("{:?}", event);
                match event {
                    glfw::WindowEvent::Key(Key::Escape, _, Action::Press, _) => {
                        window.set_should_close(true)
                    },
                    glfw::WindowEvent::Size(x,y) => {
                        uniform_size.0 = x as f32;
                        uniform_size.1 = y as f32;
                        unsafe {
                            gl.Viewport(0,0,x,y);
                        }
                    },
                    glfw::WindowEvent::MouseButton(button, action, _ ) => {
                        tmp_mouse_left_pressed = (button == MouseButton::Button1) && (action == Action::Press);
                        if tmp_mouse_left_pressed {
                            mouse.0 = tmp_mouse_pos.0 as f32;
                            mouse.1 = tmp_mouse_pos.1 as f32;
                            mouse.2 = tmp_mouse_pos.0 as f32;
                            mouse.3 = tmp_mouse_pos.1 as f32;
                        }
                        else {
                            mouse.2 = 0f32;
                            mouse.3 = 0f32;
                        }
                    },
                    glfw::WindowEvent::CursorPos(mx, my) => {
                        if mx >=0f64 && my >= 0f64 {
                            tmp_mouse_pos.0 = mx;
                            tmp_mouse_pos.1 = my;
                            if tmp_mouse_left_pressed {
                                mouse.0 = mx as f32;
                                mouse.1 = my as f32;
                                mouse.2 = mx as f32;
                                mouse.3 = my as f32;
                            }
                            else {
                                mouse.2 = 0f32;
                                mouse.3 = 0f32;
                            }
                        }
                    }
                    _ => {},
                }
            }
            //screen_size = window.get_framebuffer_size() as (f32,f32);
            unsafe {
                if self.m_mouse_uniform_loc > -1 {
                    gl.Uniform4f( self.m_mouse_uniform_loc, mouse.0, mouse.1, mouse.2, mouse.3);
                }
                if self.m_size_uniform_loc > -1 {
                    gl.Uniform2f( self.m_size_uniform_loc, uniform_size.0, uniform_size.1);
                }
                if self.m_time_uniform_loc > -1 {
                    gl.Uniform1f( self.m_time_uniform_loc, uniform_time);
                }
                gl.Clear(GL_COLOR_BUFFER_BIT);
                gl.DrawArrays(GL_TRIANGLES, 0, 6);
            }

            window.swap_buffers();

            uniform_time += now.elapsed().as_secs_f32();
        }
    }

    #[allow(temporary_cstring_as_ptr)]
    fn prepare(&mut self) -> GlFns
    {
        let gl = unsafe {
            GlFns::load_from(&|p| {
                let c_str = std::ffi::CStr::from_ptr(p as *const i8);
                let rust_str = c_str.to_str().unwrap();
                self.m_glfw.get_proc_address_raw(rust_str) as _
            }).unwrap()
        };

        unsafe {
            gl.ClearColor(0.2, 0.3, 0.3, 1.0);

            let mut vao = 0;
            gl.GenVertexArrays(1, &mut vao);
            assert_ne!(vao, 0);
            gl.BindVertexArray(vao);

            let mut vbo = 0;
            gl.GenBuffers(1, &mut vbo);
            assert_ne!(vbo, 0);
            gl.BindBuffer(GL_ARRAY_BUFFER, vbo);
            gl.BufferData(
                GL_ARRAY_BUFFER,
                size_of_val(&VERTICES) as isize,
                VERTICES.as_ptr().cast(),
                GL_STATIC_DRAW,
            );

            gl.VertexAttribPointer(
                0,
                2,
                GL_FLOAT,
                0,
                size_of::<Vertex>().try_into().unwrap(),
                0 as *const _,
            );
            gl.EnableVertexAttribArray(0);

            let vertex_shader = gl.CreateShader(GL_VERTEX_SHADER);
            assert_ne!(vertex_shader, 0);
            gl.ShaderSource(
                vertex_shader,
                1,
                &(VERT_SHADER.as_bytes().as_ptr().cast()),
                &(VERT_SHADER.len().try_into().unwrap()),
            );
            gl.CompileShader(vertex_shader);
            let mut success = 0;
            gl.GetShaderiv(vertex_shader, GL_COMPILE_STATUS, &mut success);
            if success == 0 {
                let mut v: Vec<u8> = Vec::with_capacity(1024);
                let mut log_len = 0_i32;
                gl.GetShaderInfoLog(
                    vertex_shader,
                    1024,
                    &mut log_len,
                    v.as_mut_ptr().cast(),
                );
                v.set_len(log_len.try_into().unwrap());
                panic!("Vertex Compile Error: {}", String::from_utf8_lossy(&v));
            }

            let fragment_shader = gl.CreateShader(GL_FRAGMENT_SHADER);
            assert_ne!(fragment_shader, 0);
            gl.ShaderSource(
                fragment_shader,
                1,
                &(FRAG_SHADER.as_bytes().as_ptr().cast()),
                &(FRAG_SHADER.len().try_into().unwrap()),
            );
            gl.CompileShader(fragment_shader);
            let mut success = 0;
            gl.GetShaderiv(fragment_shader, GL_COMPILE_STATUS, &mut success);
            if success == 0 {
                let mut v: Vec<u8> = Vec::with_capacity(1024);
                let mut log_len = 0_i32;
                gl.GetShaderInfoLog(
                    fragment_shader,
                    1024,
                    &mut log_len,
                    v.as_mut_ptr().cast(),
                );
                v.set_len(log_len.try_into().unwrap());
                panic!("Fragment Compile Error: {}", String::from_utf8_lossy(&v));
            }

            let shader_program = gl.CreateProgram();
            assert_ne!(shader_program, 0);
            gl.AttachShader(shader_program, vertex_shader);
            gl.AttachShader(shader_program, fragment_shader);
            gl.LinkProgram(shader_program);
            let mut success = 0;
            gl.GetProgramiv(shader_program, GL_LINK_STATUS, &mut success);
            if success == 0 {
                let mut v: Vec<u8> = Vec::with_capacity(1024);
                let mut log_len = 0_i32;
                gl.GetProgramInfoLog(
                    shader_program,
                    1024,
                    &mut log_len,
                    v.as_mut_ptr().cast(),
                );
                v.set_len(log_len.try_into().unwrap());
                panic!("Program Link Error: {}", String::from_utf8_lossy(&v));
            }
            gl.DeleteShader(vertex_shader);
            gl.DeleteShader(fragment_shader);

            self.m_size_uniform_loc = gl.GetUniformLocation(shader_program,  CString::new("size").unwrap().as_ptr() as *const u8);
            self.m_time_uniform_loc = gl.GetUniformLocation(shader_program, CString::new("time").unwrap().as_ptr() as *const u8);
            self.m_mouse_uniform_loc = gl.GetUniformLocation(shader_program, CString::new("mouse").unwrap().as_ptr() as *const u8);
            
            gl.UseProgram(shader_program);
        }

        gl
    }
}