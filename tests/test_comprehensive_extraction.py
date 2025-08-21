import unittest
from scraper.extractors import (
    extract_all_dom_features,
    extract_buttons,
    extract_input_fields,
    extract_textareas,
    extract_links,
    extract_images,
    extract_stylesheets,
    extract_scripts,
    extract_fonts,
    extract_forms,
    extract_media_resources
)


class TestComprehensiveExtraction(unittest.TestCase):
    
    def setUp(self):
        """Set up test HTML content with various DOM elements."""
        self.test_html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Test Page</title>
            <link rel="stylesheet" href="/styles.css" media="screen" integrity="sha384-..." crossorigin="anonymous">
            <link rel="preload" href="/fonts/roboto.woff2" as="font" type="font/woff2" crossorigin="anonymous">
            <style>
                @font-face {
                    font-family: 'CustomFont';
                    src: url('/fonts/custom.woff2') format('woff2');
                }
                body { font-family: Arial, sans-serif; }
            </style>
            <script src="/app.js" async defer integrity="sha384-..." crossorigin="anonymous"></script>
            <script>
                console.log('Inline script');
            </script>
        </head>
        <body>
            <header>
                <nav>
                    <a href="/" id="home-link" class="nav-link" title="Home">Home</a>
                    <a href="/about" class="nav-link">About</a>
                    <a href="https://external.com" target="_blank" rel="noopener">External</a>
                </nav>
            </header>
            
            <main>
                <h1>Test Page</h1>
                
                <form id="contact-form" name="contact" action="/submit" method="post" enctype="multipart/form-data" target="_self">
                    <input type="text" id="name" name="name" placeholder="Enter your name" required>
                    <input type="email" id="email" name="email" placeholder="Enter your email" required>
                    <input type="password" id="password" name="password" value="default">
                    <input type="checkbox" id="newsletter" name="newsletter">
                    <input type="radio" id="male" name="gender" value="male">
                    <input type="submit" value="Submit Form">
                    <input type="button" value="Reset" onclick="resetForm()">
                    
                    <textarea id="message" name="message" placeholder="Enter your message" rows="5" cols="50" required></textarea>
                    
                    <button type="submit" id="submit-btn" name="submit">Submit</button>
                    <button type="button" id="cancel-btn" name="cancel">Cancel</button>
                </form>
                
                <div class="content">
                    <img src="/hero.jpg" alt="Hero Image" id="hero-img" class="hero-image" width="800" height="400" title="Hero">
                    <img src="/logo.png" alt="Logo" class="logo">
                    
                    <video src="/video.mp4" controls autoplay width="640" height="360">
                        <source src="/video.webm" type="video/webm">
                    </video>
                    
                    <audio src="/audio.mp3" controls loop>
                        <source src="/audio.ogg" type="audio/ogg">
                    </audio>
                </div>
                
                <div class="interactive">
                    <button type="button" id="action-btn" name="action">Click Me</button>
                    <input type="text" id="search" name="search" placeholder="Search...">
                    <textarea id="notes" name="notes" placeholder="Add notes..." rows="3" cols="30"></textarea>
                </div>
            </main>
            
            <footer>
                <a href="/privacy">Privacy Policy</a>
                <a href="/terms">Terms of Service</a>
            </footer>
        </body>
        </html>
        """

    def test_extract_all_dom_features(self):
        """Test the comprehensive extraction function."""
        features = extract_all_dom_features(self.test_html)
        
        # Check that all expected keys exist
        expected_keys = ['buttons', 'inputs', 'textareas', 'links', 'images', 
                        'stylesheets', 'scripts', 'fonts', 'forms', 'media']
        for key in expected_keys:
            self.assertIn(key, features)
            self.assertIsInstance(features[key], list)

    def test_extract_buttons(self):
        """Test button extraction."""
        buttons = extract_buttons(self.test_html)
        
        # Should find 4 buttons: 2 form buttons + 2 input buttons
        self.assertEqual(len(buttons), 4)
        
        # Check specific button properties
        submit_button = next(b for b in buttons if b.get('text') == 'Submit Form')
        self.assertEqual(submit_button['type'], 'submit')
        
        action_button = next(b for b in buttons if b.get('id') == 'action-btn')
        self.assertEqual(action_button['text'], 'Click Me')
        self.assertEqual(action_button['type'], 'button')

    def test_extract_input_fields(self):
        """Test input field extraction."""
        inputs = extract_input_fields(self.test_html)
        
        # Should find 8 input fields
        self.assertEqual(len(inputs), 8)
        
        # Check specific input types
        text_inputs = [i for i in inputs if i['type'] == 'text']
        self.assertEqual(len(text_inputs), 2)
        
        email_input = next(i for i in inputs if i['type'] == 'email')
        self.assertEqual(email_input['name'], 'email')
        self.assertEqual(email_input['placeholder'], 'Enter your email')
        self.assertTrue(email_input['required'])

    def test_extract_textareas(self):
        """Test textarea extraction."""
        textareas = extract_textareas(self.test_html)
        
        # Should find 2 textareas
        self.assertEqual(len(textareas), 2)
        
        message_ta = next(t for t in textareas if t['id'] == 'message')
        self.assertEqual(message_ta['rows'], '5')
        self.assertEqual(message_ta['cols'], '50')
        self.assertTrue(message_ta['required'])

    def test_extract_links(self):
        """Test link extraction."""
        links = extract_links(self.test_html)
        
        # Should find 7 links
        self.assertEqual(len(links), 7)
        
        # Check internal vs external links
        internal_links = [l for l in links if not l['href'].startswith('http')]
        external_links = [l for l in links if l['href'].startswith('http')]
        
        self.assertEqual(len(internal_links), 6)
        self.assertEqual(len(external_links), 1)
        
        # Check specific link properties
        home_link = next(l for l in links if l['id'] == 'home-link')
        self.assertEqual(home_link['class'], 'nav-link')
        self.assertEqual(home_link['title'], 'Home')

    def test_extract_images(self):
        """Test image extraction."""
        images = extract_images(self.test_html)
        
        # Should find 2 images
        self.assertEqual(len(images), 2)
        
        hero_img = next(i for i in images if i['id'] == 'hero-img')
        self.assertEqual(hero_img['src'], '/hero.jpg')
        self.assertEqual(hero_img['alt'], 'Hero Image')
        self.assertEqual(hero_img['width'], '800')
        self.assertEqual(hero_img['height'], '400')

    def test_extract_stylesheets(self):
        """Test stylesheet extraction."""
        stylesheets = extract_stylesheets(self.test_html)
        
        # Should find 2 stylesheets (1 external + 1 inline)
        self.assertEqual(len(stylesheets), 2)
        
        external_css = next(s for s in stylesheets if s['type'] == 'stylesheet')
        self.assertEqual(external_css['url'], '/styles.css')
        self.assertEqual(external_css['media'], 'screen')
        self.assertTrue(external_css['integrity'].startswith('sha384-'))
        
        inline_css = next(s for s in stylesheets if s['type'] == 'inline_style')
        self.assertEqual(inline_css['url'], 'inline')
        self.assertIn('font-family', inline_css['content'])

    def test_extract_scripts(self):
        """Test script extraction."""
        scripts = extract_scripts(self.test_html)
        
        # Should find 2 scripts (1 external + 1 inline)
        self.assertEqual(len(scripts), 2)
        
        external_js = next(s for s in scripts if s['type'] == 'external_script')
        self.assertEqual(external_js['url'], '/app.js')
        self.assertTrue(external_js['async'])
        self.assertTrue(external_js['defer'])
        
        inline_js = next(s for s in scripts if s['type'] == 'inline_script')
        self.assertEqual(inline_js['url'], 'inline')
        self.assertIn('console.log', inline_js['content'])

    def test_extract_fonts(self):
        """Test font extraction."""
        fonts = extract_fonts(self.test_html)
        
        # Should find 2 fonts (1 preload + 1 CSS declared)
        self.assertEqual(len(fonts), 2)
        
        preload_font = next(f for f in fonts if f['format'] == 'font/woff2')
        self.assertEqual(preload_font['url'], '/fonts/roboto.woff2')
        self.assertEqual(preload_font['type'], 'font')
        
        css_font = next(f for f in fonts if f['format'] == 'css_declared')
        self.assertEqual(css_font['url'], '/fonts/custom.woff2')

    def test_extract_forms(self):
        """Test form extraction."""
        forms = extract_forms(self.test_html)
        
        # Should find 1 form
        self.assertEqual(len(forms), 1)
        
        form = forms[0]
        self.assertEqual(form['id'], 'contact-form')
        self.assertEqual(form['name'], 'contact')
        self.assertEqual(form['action'], '/submit')
        self.assertEqual(form['method'], 'post')
        self.assertEqual(form['enctype'], 'multipart/form-data')
        
        # Check form inputs
        self.assertEqual(len(form['inputs']), 8)
        self.assertEqual(len(form['buttons']), 2)

    def test_extract_media_resources(self):
        """Test media resource extraction."""
        media = extract_media_resources(self.test_html)
        
        # Should find 3 media resources (1 video + 1 audio + 1 source)
        self.assertEqual(len(media), 3)
        
        video = next(m for m in media if m['type'] == 'video')
        self.assertEqual(video['url'], '/video.mp4')
        self.assertEqual(video['width'], '640')
        self.assertEqual(video['height'], '360')
        self.assertTrue(video['controls'])
        self.assertTrue(video['autoplay'])
        
        audio = next(m for m in media if m['type'] == 'audio')
        self.assertEqual(audio['url'], '/audio.mp3')
        self.assertTrue(audio['controls'])
        self.assertTrue(audio['loop'])

    def test_comprehensive_data_structure(self):
        """Test that all extracted data has consistent structure."""
        features = extract_all_dom_features(self.test_html)
        
        # Test that all features have expected structure
        for feature_type, feature_list in features.items():
            self.assertIsInstance(feature_list, list)
            
            for feature in feature_list:
                self.assertIsInstance(feature, dict)
                # Each feature should have at least one property
                self.assertGreater(len(feature), 0)


if __name__ == '__main__':
    unittest.main()
